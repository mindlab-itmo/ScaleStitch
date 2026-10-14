# Adapted from Graph3DLLM / Chat-Scene; see LICENSE.host.
import random
import logging
from abc import ABC

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch.nn.functional as F
import einops


from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
from .position_embedding import PositionEmbeddingCoordsSine
from peft import LoraConfig, get_peft_model

from torch.nn.utils.rnn import pad_sequence

import contextlib
from .data import recover_caption

logger = logging.getLogger(__name__)


def _config_value(node, name, default=None):
    if node is None:
        return default
    getter = getattr(node, "get", None)
    if callable(getter):
        return getter(name, default)
    return getattr(node, name, default)


def _config_keys(node):
    if node is None:
        return set()
    keys = getattr(node, "keys", None)
    if callable(keys):
        return set(keys())
    return None


def preferred_cuda_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    is_supported = getattr(torch.cuda, "is_bf16_supported", None)
    if callable(is_supported) and is_supported():
        return torch.bfloat16
    return torch.float16


def _generation_prefix_attention_mask(inputs_embeds):
    if not isinstance(inputs_embeds, torch.Tensor) or inputs_embeds.ndim != 3:
        raise ValueError("generation inputs_embeds must have shape [batch, tokens, dim]")
    return torch.ones(
        inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device
    )


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False

def nclamp(input, min, max):
    return input.clamp(min=min, max=max).detach() + input - input.detach()


def print_grad_status(model):
    for name, p in model.named_parameters():
        print('{:80s}{:20s}{:20s}{}'.format(name,
            '(Trainable)' if p.requires_grad else '(Fixed)',
            '(Has grad):' if p.grad is not None else '(No grad backward):',
            list(p.shape)))


class Graph3DLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        llama_model_path = config.model.llama_model_path
        self.low_resource = config.model.low_resource
        self.max_txt_len = config.model.max_txt_len
        self.end_sym = config.model.end_sym
        self.system = config.model.system
        self.instruction = config.model.instruction
        self.role = config.model.role
        self.no_obj = config.model.no_obj
        if self.no_obj:
            logger.warning(
                "The retained no_obj flag does not remove object features; "
                "use model.object_feature_intervention for an explicit control"
            )
        self.add_scene_token = config.model.add_scene_token
        self.add_img_token = config.model.add_img_token
        self.train_emb = config.model.train_emb
        self.train_img_proj = config.model.train_img_proj
        self.input_dim = config.model.input_dim
        self.source_feature_dim = _config_value(
            config.model, "source_feature_dim", self.input_dim
        )
        if (
            not isinstance(self.source_feature_dim, int)
            or isinstance(self.source_feature_dim, bool)
            or self.source_feature_dim <= 0
        ):
            raise ValueError("model.source_feature_dim must be a positive integer")
        self.img_input_dim = config.model.img_input_dim
        self.attr_dim = config.model.attr_dim
        self.scene_dim = config.model.scene_dim
        self.pos_dim = config.model.pos_dim
        self.max_obj_num = config.model.max_obj_num
        self.bidirection = config.model.bidirection
        self.add_pos_emb = config.model.add_pos_emb
        self.feat_fusion = config.model.feat_fusion
        self.fuse_with_id = config.model.fuse_with_id
        self.use_location_token = config.model.use_location_token
        self.knn = config.model.knn
        self.max_knn = config.model.max_knn
        self.bbox_embed = config.model.bbox_embed
        self.gt_pretrain = config.model.gt_pretrain
        self.nms = config.model.nms
        self.nn_distance = config.model.nn_distance
        self.host_mode = _config_value(config.model, "host_mode", "legacy")
        self.missing_channel_policy = _config_value(config.model, "missing_channel_policy", "reject")
        self._declared_missing_channels()
        self.object_feature_intervention = _config_value(
            config.model, "object_feature_intervention", "full"
        )
        if self.host_mode not in {"legacy", "full", "slot_null", "object_only"}:
            raise ValueError(
                "model.host_mode must be one of legacy/full/slot_null/object_only"
            )
        if self.host_mode != "legacy" and (self.add_scene_token or self.bbox_embed):
            raise ValueError(
                "Non-legacy protocols reject the currently non-functional "
                "add_scene_token and bbox_embed paths"
            )
        if self.host_mode == "object_only":
            incompatible = {
                "add_img_token": self.add_img_token,
                "add_scene_token": self.add_scene_token,
                "knn": self.knn,
                "bbox_embed": self.bbox_embed,
            }
            if self.add_img_token or self.add_scene_token or self.knn != 0 or self.bbox_embed:
                raise ValueError(
                    "object_only requires add_img_token=False, add_scene_token=False, "
                    f"knn=0, bbox_embed=False; got {incompatible}"
                )
        if self.host_mode == "full" and (
            not self.add_img_token or self.img_input_dim != 1024
            or type(self.knn) is not int or self.knn < 1
            or type(self.max_knn) is not int or self.max_knn < self.knn
        ):
            raise ValueError("full requires 1024-wide image inputs and 1 <= knn <= max_knn")
        self._validate_object_feature_intervention()

        self.debug = config.debug
        if not self.debug:
            logger.info('Loading LLaMA')

            llm_dtype = preferred_cuda_dtype()
            if "vicuna" in llama_model_path or self.low_resource:
                raise ValueError("These experiments use the unquantized Llama-3 host")
            self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_path, use_fast=False)
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
            self.llama3 = False
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                llama_model_path, torch_dtype=llm_dtype, attn_implementation="sdpa")


            logger.info("freeze LLAMA")
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False

            if config.model.use_lora:
                def find_linear_layers(model, lora_target_modules):
                    cls = torch.nn.Linear
                    lora_module_names = set()
                    for name, module in model.named_modules():
                        if (
                            isinstance(module, cls)
                            and all(
                                [
                                    x not in name
                                    for x in [
                                        "instance2embed",
                                        "hidden_state2query"
                                    ]
                                ]
                            )
                            and any([x in name for x in lora_target_modules])
                        ):
                            lora_module_names.add(name)
                    return sorted(list(lora_module_names))

                lora_target_modules = find_linear_layers(self.llama_model, config.lora.lora_target_modules)

                lora_config = LoraConfig(
                    r=config.lora.lora_r,
                    lora_alpha=config.lora.lora_alpha,
                    target_modules=lora_target_modules,
                    lora_dropout=config.lora.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                self.llama_model = get_peft_model(self.llama_model, lora_config)
                self.llama_model.print_trainable_parameters()
                if self.train_emb:
                    self.llama_model.model.lm_head.weight.requires_grad = True
                    self.llama_model.model.lm_head.weight.data = self.llama_model.model.lm_head.weight.data.float()
                    self.llama_model.print_trainable_parameters()
                    self.llama_model.model.model.embed_tokens.weight.requires_grad = True
                    self.llama_model.model.model.embed_tokens.weight.data = (
                        self.llama_model.model.model.embed_tokens.weight.data.float()
                    )
                    self.llama_model.print_trainable_parameters()
            else:
                if self.train_emb:
                    self.llama_model.lm_head.weight.requires_grad = True
                    self.llama_model.lm_head.weight.data = self.llama_model.lm_head.weight.data.float()
                    self.llama_model.model.embed_tokens.weight.requires_grad = True
                    self.llama_model.model.embed_tokens.weight.data = (
                        self.llama_model.model.embed_tokens.weight.data.float()
                    )

            self.llama_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
            objid_tokens = []
            for i in range(self.max_obj_num):
                objid_tokens.append(f"<OBJ{i:03}>")
            self.objid_start_idx = self.ori_vocab_size = len(self.llama_tokenizer)
            self.llama_tokenizer.add_tokens(objid_tokens, special_tokens=True)
            self.objid_end_idx = len(self.llama_tokenizer)
            self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))


            self.llama_dim = self.llama_model.config.hidden_size
            logger.info('Loading LLAMA Done')
        else:
            self.llama_model = None
            self.llama_dim = 4096


        self.object_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.llama_dim),
            nn.GELU(),
            nn.Linear(self.llama_dim, self.llama_dim)
        )
        self.object_img_proj = nn.Sequential(
            nn.Linear(self.img_input_dim, self.llama_dim),
            nn.GELU(),
            nn.Linear(self.llama_dim, self.llama_dim)
        )

        if self.bbox_embed:
            self.coord_proj = nn.Sequential(
                nn.Linear(6, 512),


            )

            self.edge_proj = nn.Sequential(
                nn.Linear(2*self.input_dim+1024, self.llama_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.LayerNorm(self.llama_dim),
                nn.Linear(self.llama_dim, self.llama_dim)
            )
        else:
            self.edge_proj = nn.Sequential(
                nn.Linear(512, self.llama_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.LayerNorm(self.llama_dim),
                nn.Linear(self.llama_dim, self.llama_dim)
            )
        if not self.train_img_proj:
            for p in self.object_img_proj.parameters():
                p.requires_grad = False
        self.pos_embedding = PositionEmbeddingCoordsSine(d_pos=self.pos_dim)
        self.pos_proj = nn.Sequential(
            nn.Linear(self.pos_dim, self.llama_dim)
        )
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.scene_dim, nhead=8, dim_feedforward=2048, dropout=0.05, norm_first=True, batch_first=True)
        self.relation_module = nn.TransformerEncoder(self.encoder_layer, num_layers=config.model.encoder_num_layers)
        self.scene_init_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.scene_dim)
        )
        self.scene_proj = nn.Sequential(
            nn.Linear(self.scene_dim, self.llama_dim),


        )

        if not self.add_scene_token:
            for p in self.relation_module.parameters():
                p.requires_grad = False
            for p in self.scene_init_proj.parameters():
                p.requires_grad = False
            for p in self.scene_proj.parameters():
                p.requires_grad = False


        if not self.debug:
            self.p_0_embed, self.p_1_embed = self.prepare_fixed_embed()
        self.last_embed = None
        self.scalestitch = None
        self._scalestitch_deployment_bundle = False
        self._scalestitch_freeze_host = False
        self.metric_quotient_token = None
        self._metric_quotient_host_finalized = False

    def get_objid_embeds(self):
        if self.config.model.use_lora:
            objid_embeds = self.llama_model.model.model.embed_tokens.weight[self.objid_start_idx:self.objid_end_idx]
        else:
            objid_embeds = self.llama_model.model.embed_tokens.weight[self.objid_start_idx:self.objid_end_idx]
        return objid_embeds

    def _llama_embed_device(self):
        if self.config.model.use_lora:
            return self.llama_model.model.model.embed_tokens.weight.device
        return self.llama_model.model.embed_tokens.weight.device

    def llama_embed_tokens(self, token_ids):
        device = self._llama_embed_device()
        if token_ids.device != device:
            token_ids = token_ids.to(device)
        if self.config.model.use_lora:
            return self.llama_model.model.model.embed_tokens(token_ids)
        else:
            return self.llama_model.model.embed_tokens(token_ids)

    def prepare_fixed_embed(self):
        prompt = self.system + " " + self.instruction + " " + self.role[0] + ": "
        p_0, p_1 = prompt.split("<REPLACE>")
        p_0_token = self.llama_tokenizer(p_0, return_tensors="pt", add_special_tokens=True)
        p_1_token = self.llama_tokenizer(p_1, return_tensors="pt", add_special_tokens=False)
        p_0_embed = self.llama_embed_tokens(p_0_token.input_ids).squeeze(0).detach()
        p_1_embed = self.llama_embed_tokens(p_1_token.input_ids).squeeze(0).detach()
        return p_0_embed, p_1_embed

    def get_text_emb(self, text, device="cpu"):
        text_tokens = self.llama_tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
        embeds = self.llama_embed_tokens(text_tokens.input_ids)
        if self.train_emb:
            indices = text_tokens.input_ids >= self.ori_vocab_size
            indices = (indices * 1).unsqueeze(-1)
            embeds = (1 - indices) * embeds.detach() + indices * embeds
        else:
            embeds = embeds.detach()
        return embeds


    def _frozen_text_embedding(self, text, expected_token_ids):
        if not isinstance(text, str) or not text:
            raise ValueError("metric quotient token text must be non-empty")
        tokenized = self.llama_tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        )
        token_ids = [int(value) for value in tokenized.input_ids.reshape(-1).tolist()]
        expected = [int(value) for value in expected_token_ids]
        if token_ids != expected:
            raise RuntimeError(
                "metric quotient tokenizer identity mismatch for "
                f"{text!r}: expected={expected}, observed={token_ids}"
            )
        embeddings = self.llama_embed_tokens(tokenized.input_ids).squeeze(0).detach()
        if (
            embeddings.ndim != 2
            or embeddings.shape[0] != len(expected)
            or embeddings.shape[1] != self.llama_dim
            or not torch.isfinite(embeddings).all()
        ):
            raise RuntimeError("metric quotient lexical embedding is malformed")
        return token_ids, embeddings.to(torch.float32)


    def train(self, mode=True):
        if getattr(self, "_scalestitch_freeze_host", False):
            nn.Module.train(self, False)
            if self.scalestitch is not None:
                self.scalestitch.train(
                    mode
                    and getattr(self, "_scalestitch_execution_mode", None)
                    in {None, "adapter_training"}
                )
            if getattr(self, "metric_quotient_token", None) is not None:
                self.metric_quotient_token.train(
                    mode
                    and getattr(self, "_metric_quotient_execution_mode", None)
                    == "adapter_training"
                )
            return self
        return super().train(mode)

    def encode_object_feat(
        self,
        feat,
        img_feat,
        locs,
        object_radius=None,
        radius_valid=None,
        scene_mask=None,
    ):
        if self.scalestitch is not None and not torch.isfinite(feat).all():
            raise FloatingPointError("ScaleStitch received non-finite object features")
        img_feat = torch.nan_to_num(torch.nn.functional.normalize(img_feat, dim=-1))
        if not getattr(self, "_scalestitch_deployment_bundle", False):
            feat = torch.nan_to_num(torch.nn.functional.normalize(feat, dim=-1))
        if self.scalestitch is not None:
            if object_radius is None or radius_valid is None or scene_mask is None:
                raise ValueError(
                    "ScaleStitch requires scene_radius, radius_valid, and scene_mask"
                )
            leading_shape = feat.shape[:-1]
            expected = feat.numel() // feat.shape[-1]
            radius_tensor = torch.as_tensor(object_radius, device=feat.device)
            radius_valid_tensor = torch.as_tensor(radius_valid, device=feat.device)
            scene_mask_tensor = torch.as_tensor(scene_mask, device=feat.device)
            if (
                radius_tensor.numel() != expected
                or radius_valid_tensor.numel() != expected
                or scene_mask_tensor.numel() != expected
            ):
                raise ValueError("ScaleStitch radius/mask shapes must match object rows")
            radius_tensor = radius_tensor.reshape(leading_shape)
            combined_valid = radius_valid_tensor.reshape(leading_shape).to(
                dtype=torch.bool
            ) & scene_mask_tensor.reshape(leading_shape).to(dtype=torch.bool)
            active_radius = radius_tensor[combined_valid]
            if active_radius.numel() and (
                not torch.isfinite(active_radius).all()
                or torch.any(active_radius <= self.scalestitch.radius_epsilon)
            ):
                raise ValueError(
                    "Active protocol objects require finite positive exact radii"
                )
            feat = self.scalestitch(feat, radius_tensor, combined_valid)
            if getattr(self, "_scalestitch_deployment_bundle", False):


                scene_active = scene_mask_tensor.reshape(leading_shape).to(
                    dtype=torch.bool
                )
                feat = torch.where(scene_active.unsqueeze(-1), feat, torch.zeros_like(feat))
        elif getattr(self, "_scalestitch_deployment_bundle", False):
            raise RuntimeError("deployment bundle flag set without a loaded pipeline")
        return feat, img_feat

    @staticmethod
    def _module_dtype(module):
        return next(module.parameters()).dtype

    def _cast_scene_inputs(self, object_embed, object_img_embed, scene_gnn_feats, scene_locs):
        object_dtype = self._module_dtype(self.object_proj)
        img_dtype = self._module_dtype(self.object_img_proj)
        edge_dtype = self._module_dtype(self.edge_proj)

        object_embed = object_embed.to(dtype=object_dtype)
        object_img_embed = object_img_embed.to(dtype=img_dtype)
        scene_gnn_feats = scene_gnn_feats.to(dtype=torch.float32 if self.host_mode == "full" else edge_dtype)
        # Do not round coordinates to the language/projector dtype before KNN.
        scene_locs = scene_locs.to(dtype=torch.float32 if self.host_mode == "full" else object_dtype)
        return object_embed, object_img_embed, scene_gnn_feats, scene_locs

    def _apply_host_mode(self, scene_img_feat, scene_gnn_feats, scene_mask):
        if self.host_mode == "legacy":
            return scene_img_feat, scene_gnn_feats
        if not torch.isfinite(scene_img_feat).all() or not torch.isfinite(scene_gnn_feats).all():
            raise FloatingPointError("Non-legacy host modes reject non-finite auxiliary features")
        if self.host_mode == "slot_null":
            return torch.zeros_like(scene_img_feat), torch.zeros_like(scene_gnn_feats)
        if self.host_mode == "object_only":
            return scene_img_feat, scene_gnn_feats


        valid = scene_mask.to(dtype=torch.bool)
        if (scene_img_feat.shape != (*scene_mask.shape, 1024)
                or not scene_img_feat.is_floating_point() or not scene_gnn_feats.is_floating_point()):
            raise ValueError("full requires floating 1024-wide images and raw floating relations")
        image_norm = torch.linalg.vector_norm(scene_img_feat.float(), dim=-1)
        if valid.any() and torch.any(image_norm[valid] == 0) and not self._declared_missing_channels():
            raise ValueError("full host_mode contains zero/missing image features")
        if (scene_gnn_feats.ndim != 4 or scene_gnn_feats.shape[:3] !=
                (scene_mask.shape[0], scene_mask.shape[1], scene_mask.shape[1])
                or scene_gnn_feats.shape[-1] != 512):
            raise ValueError("full requires a raw [batch, source row, target row, 512] relation bank")
        if torch.any(valid.sum(dim=1) <= self.knn):
            raise ValueError("full requires more active objects than knn; self-relation fallback is forbidden")
        return scene_img_feat, scene_gnn_feats

    def _declared_missing_channels(self):
        policy = getattr(self, "missing_channel_policy", "reject")
        if policy not in {"reject", "zero_projected_declared"}:
            raise ValueError("invalid model.missing_channel_policy")
        if policy != "reject" and self.host_mode != "full":
            raise ValueError("declared missing channels require host_mode=full")
        return policy == "zero_projected_declared"

    def _zero_declared_image_slots(self, projected, raw_images):
        # Raw source rows are checked before normalization/casts/ID assignment.
        # Call after both projector bias and optional position addition.
        if projected is not None and self._declared_missing_channels():
            missing = torch.count_nonzero(raw_images, dim=-1) == 0
            projected = projected.masked_fill(missing.unsqueeze(-1), 0)
        return projected

    def _select_full_relations(self, bank, source_rows, target_rows):
        """Project only relations selected by actual source/target row identity.

        The dense raw bank uses zero vectors exclusively for unavailable pairs.
        A missing selected pair is an input error, never a projected zero slot.
        """
        if (bank.ndim != 3 or bank.shape[0] != bank.shape[1] or bank.shape[2] != 512
                or source_rows.ndim != 1 or target_rows.ndim != 2
                or target_rows.shape != (len(source_rows), self.knn)):
            raise ValueError("malformed full relation bank or selected row identities")
        if (torch.any(source_rows[:, None] == target_rows)
                or torch.any(source_rows < 0) or torch.any(source_rows >= bank.shape[0])
                or torch.any(target_rows < 0) or torch.any(target_rows >= bank.shape[1])):
            raise ValueError("full relation requires distinct, valid source and target rows")
        selected = bank[source_rows[:, None], target_rows].float()
        missing = torch.count_nonzero(selected, dim=-1) == 0
        if not torch.isfinite(selected).all() or (torch.any(missing) and not self._declared_missing_channels()):
            raise ValueError("selected source/target relation is missing or nonfinite; zero imputation forbidden")
        normalized = F.normalize(selected, dim=-1).to(dtype=self._module_dtype(self.edge_proj))
        projected = self.edge_proj(normalized)
        if not torch.isfinite(projected).all():
            raise FloatingPointError("nonfinite projected full relation")
        if self._declared_missing_channels():
            projected = projected.masked_fill(missing.unsqueeze(-1), 0)
        return projected

    @staticmethod
    def get_dist_attention(pos, dist_exp=1):

        dist = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = torch.sum(dist.abs()**dist_exp, dim=-1)
        dist_attn = torch.nn.functional.softmax(-dist, dim=-1)
        return dist_attn

    def _validate_object_feature_intervention(self):
        mode = getattr(self, "object_feature_intervention", "full")
        choices = {"full", "zero_object_slots", "drop_object_features", "permute_object_slots"}
        if mode not in choices:
            raise ValueError(f"model.object_feature_intervention must be one of {sorted(choices)}")
        if mode != "full" and (
            self.host_mode != "object_only" or self.knn != 0 or self.add_img_token
            or self.add_scene_token or self.bbox_embed
        ):
            raise ValueError("object feature controls require the object-only host without auxiliary tokens")
        return mode

    def _object_feature_slots(self, embed_obj, assigned_ids):
        """Intervene on projected object slots, including any positional component.

        Object-ID and metric tokens remain bound to their original objects.
        Permutation is a cyclic shift of sorted source rows, independent of the
        presented ID order. These controls remove mixed object information, not
        an isolated geometric or semantic component of an encoder.
        """
        mode = self._validate_object_feature_intervention()
        if mode == "drop_object_features":
            return None
        selected = embed_obj[assigned_ids]
        if mode == "zero_object_slots":
            return torch.zeros_like(selected)
        if mode == "permute_object_slots":
            source_ids = torch.sort(assigned_ids).values
            if source_ids.numel() < 2 or torch.unique(source_ids).numel() != source_ids.numel():
                raise ValueError("object permutation requires at least two distinct active objects")
            permuted_ids = torch.roll(source_ids, shifts=1)
            selected = embed_obj[permuted_ids[torch.searchsorted(source_ids, assigned_ids)]]
        return selected

    def _get_object_list_embed_protocol(
        self,
        embed_obj,
        embed_img,
        scene_mask,
        assigned_ids,
        proj_edge_embed,
        scene_locs,
        foreground_ids,
        metric_token_embed=None,
    ):
        mode = self._validate_object_feature_intervention()
        valid_ids = torch.where(scene_mask.to(dtype=torch.bool))[0]
        assigned_ids = assigned_ids.to(valid_ids.device)[valid_ids].to(
            device=embed_obj.device, dtype=torch.long
        )
        if mode != "full" or self.host_mode == "full":
            active_sources = scene_mask.to(device=assigned_ids.device, dtype=torch.bool)
            if (
                torch.any(assigned_ids < 0)
                or torch.any(assigned_ids >= active_sources.numel())
                or torch.any(~active_sources[assigned_ids])
            ):
                raise ValueError("object feature controls require every visible slot to reference an active source row")
            if torch.unique(assigned_ids).numel() != assigned_ids.numel():
                raise ValueError("visible slots must reference distinct source rows")
        all_objid_embeds = self.get_objid_embeds()
        objid_embeds = all_objid_embeds[valid_ids.to(all_objid_embeds.device)]
        foreground_ids = foreground_ids.to(device=assigned_ids.device, dtype=torch.long)
        if self.gt_pretrain:
            foreground_ids = foreground_ids[valid_ids.to(foreground_ids.device)]

        if self.knn == 0:
            object_slots = self._object_feature_slots(embed_obj, assigned_ids)
            blocks = []
            for row, assigned_id in enumerate(assigned_ids):
                tokens = [objid_embeds[row]]
                if object_slots is not None:
                    tokens.append(object_slots[row])
                if metric_token_embed is not None:
                    metric_block = metric_token_embed[assigned_id]
                    if (
                        metric_block.ndim != 2
                        or metric_block.shape[1] != embed_obj.shape[1]
                        or not torch.isfinite(metric_block).all()
                    ):
                        raise ValueError("metric token block is malformed or non-finite")
                    tokens.extend(metric_block.unbind(dim=0))
                if self.add_img_token:
                    if embed_img is None:
                        raise RuntimeError("add_img_token=True requires image embeddings")
                    tokens.append(embed_img[assigned_id])
                blocks.append(torch.stack(tokens, dim=0))
            if not blocks:
                return objid_embeds.new_zeros((0, objid_embeds.shape[-1]))
            return torch.cat(blocks, dim=0).to(dtype=objid_embeds.dtype)

        if metric_token_embed is not None:
            raise RuntimeError("metric quotient token v1 does not support knn>0")
        if proj_edge_embed is None:
            raise RuntimeError("knn>0 requires concrete projected edge embeddings")
        foreground_set = {int(value) for value in foreground_ids.tolist()}
        foreground_assigned_ids = torch.as_tensor(
            [int(value) for value in assigned_ids.tolist() if int(value) in foreground_set],
            dtype=torch.long,
            device=assigned_ids.device,
        )
        if not self.gt_pretrain and self.nms:
            candidate_ids = foreground_assigned_ids
        else:
            candidate_ids = assigned_ids
        if candidate_ids.numel() == 0:
            raise RuntimeError("No foreground neighbor candidates in protocol host mode")

        if self.host_mode == "full":
            # Sorting makes exact-distance ties independent of presentation order.
            candidate_ids = candidate_ids.sort().values

        query_locs = scene_locs[assigned_ids, :3]
        candidate_locs = scene_locs[candidate_ids, :3]
        pairwise_dists = torch.cdist(query_locs.float(), candidate_locs.float())
        if self.host_mode == "full":
            if candidate_ids.numel() <= self.knn or not torch.isfinite(pairwise_dists).all():
                raise ValueError("full requires finite geometry and more neighbor candidates than knn")
            unavailable = assigned_ids[:, None] == candidate_ids[None, :]
            if self.gt_pretrain or self.nn_distance:
                unavailable = unavailable | (pairwise_dists < 0.01)
            pairwise_dists = pairwise_dists.masked_fill(unavailable, float("inf"))
            if torch.any(torch.isfinite(pairwise_dists).sum(dim=1) < self.knn):
                raise ValueError("full lacks enough distinct geometric neighbors; self fallback is forbidden")
            columns = torch.argsort(pairwise_dists, dim=1, stable=True)[:, :self.knn]
            target_rows = candidate_ids[columns]
            selected_edges = self._select_full_relations(proj_edge_embed, assigned_ids, target_rows)
            if embed_img is None:
                raise ValueError("full requires projected image features")
            blocks = []
            for row, assigned_id in enumerate(assigned_ids):
                tokens = [objid_embeds[row], embed_img[assigned_id]]
                for neighbour in range(self.knn):
                    tokens.extend((embed_obj[assigned_id], selected_edges[row, neighbour],
                                   embed_obj[target_rows[row, neighbour]]))
                blocks.append(torch.stack(tokens))
            return torch.cat(blocks).to(dtype=objid_embeds.dtype)
        if self.gt_pretrain or self.nn_distance:
            pairwise_dists = pairwise_dists.masked_fill(pairwise_dists < 0.01, 100.0)
            neighbour_shift = 0
        else:
            neighbour_shift = 1

        enough_distinct = candidate_ids.numel() > self.knn
        topk_indices = None
        if enough_distinct:
            topk_indices = torch.topk(
                pairwise_dists,
                self.knn + neighbour_shift,
                dim=1,
                largest=False,
            ).indices

        blocks = []
        for row, assigned_id in enumerate(assigned_ids):
            tokens = [objid_embeds[row]]
            if self.add_img_token:
                if embed_img is None:
                    raise RuntimeError("add_img_token=True requires image embeddings")
                tokens.append(embed_img[assigned_id])
            for neighbour in range(self.knn):
                edge_id = self.max_knn * int(assigned_id) + neighbour
                if edge_id >= proj_edge_embed.shape[0]:
                    raise RuntimeError(
                        f"edge feature array does not cover required index {edge_id}"
                    )
                neighbour_id = assigned_id
                if enough_distinct:
                    candidate_column = topk_indices[row, neighbour + neighbour_shift]
                    neighbour_id = candidate_ids[candidate_column]
                tokens.extend(
                    [
                        embed_obj[assigned_id],
                        proj_edge_embed[edge_id],
                        embed_obj[neighbour_id],
                    ]
                )
            blocks.append(torch.stack(tokens, dim=0))
        if not blocks:
            return objid_embeds.new_zeros((0, objid_embeds.shape[-1]))
        return torch.cat(blocks, dim=0).to(dtype=objid_embeds.dtype)

    def get_object_list_embed(self, embed_obj, embed_img, embed_scene, scene_mask, obj_id, assigned_ids,               proj_edge_embed, scene_locs, foreground_ids, scene_feat, metric_token_embed=None):
        if self.host_mode != "legacy":
            return self._get_object_list_embed_protocol(
                embed_obj,
                embed_img,
                scene_mask,
                assigned_ids,
                proj_edge_embed,
                scene_locs,
                foreground_ids,
                metric_token_embed,
            )
        if metric_token_embed is not None:
            raise RuntimeError("metric quotient token v1 does not support legacy host mode")
        valid_ids = torch.where(scene_mask)[0].tolist()
        if self.config.model.use_lora:
            objid_embeds = self.llama_model.model.model.embed_tokens.weight[self.objid_start_idx:self.objid_end_idx]
        else:
            objid_embeds = self.llama_model.model.embed_tokens.weight[self.objid_start_idx:self.objid_end_idx]

        assigned_ids = assigned_ids[valid_ids]
        selected_objid_embeds = objid_embeds[valid_ids]

        foreground_ids = foreground_ids.to(selected_objid_embeds.device)

        if self.gt_pretrain:
            foreground_ids = foreground_ids[valid_ids]
        if self.knn > 0:
            object_list_embed = torch.zeros((selected_objid_embeds.shape[0] * (3 * self.knn + 2), selected_objid_embeds.shape[1]), dtype=selected_objid_embeds.dtype, device=selected_objid_embeds.device)

            edges_assigned_ids = []
            foreground_assigned_ids = []
            for assigned_id in assigned_ids:
                if assigned_id in foreground_ids:
                    foreground_assigned_ids.append(int(assigned_id))
                for nn in range(self.knn):
                    edges_assigned_ids.append(self.max_knn*int(assigned_id)+nn)

            if not self.gt_pretrain and self.nms:
                pairwise_locs = einops.repeat(scene_locs[assigned_ids, :3], 'l d -> l 1 d')\
                    - einops.repeat(scene_locs[foreground_assigned_ids, :3], 'l d -> 1 l d')
                pairwise_dists = torch.sqrt(torch.sum(pairwise_locs ** 2, 2) + 1e-10)
            else:
                pairwise_locs = einops.repeat(scene_locs[assigned_ids, :3], 'l d -> l 1 d')\
                    - einops.repeat(scene_locs[assigned_ids, :3], 'l d -> 1 l d')
                pairwise_dists = torch.sqrt(torch.sum(pairwise_locs ** 2, 2) + 1e-10)


            if self.gt_pretrain or self.nn_distance:
                MINIMUM_DISTANCE = 0.01
                pairwise_dists[pairwise_dists < MINIMUM_DISTANCE] = 100.0
                NEIGHBOR_SHIFT = 0
            else:
                NEIGHBOR_SHIFT = 1
            obj_num = selected_objid_embeds.shape[0]
            if self.knn < obj_num:
                topk_values, topk_indices = torch.topk(pairwise_dists, self.knn+NEIGHBOR_SHIFT, dim=1,  largest=False)

                object_list_embed[0::(3*self.knn+2), :] = selected_objid_embeds.to(object_list_embed.dtype)
                object_list_embed[1::(3*self.knn+2), :] = embed_img[assigned_ids]
                for nn in range(self.knn):
                    object_list_embed[2 + 3*nn::(3*self.knn+2), :] = embed_obj[assigned_ids]
                    if max(edges_assigned_ids) < proj_edge_embed.shape[0]:


                        proj_edge_embed = proj_edge_embed[edges_assigned_ids]
                        object_list_embed[3 + 3*nn::(3*self.knn+2), :] = proj_edge_embed[nn::self.knn,:].to(object_list_embed.dtype)
                    if not self.gt_pretrain and self.nms:
                        object_list_embed[4 + 3*nn::(3*self.knn+2), :] = embed_obj[torch.tensor(foreground_assigned_ids)[topk_indices[:,nn+NEIGHBOR_SHIFT].cpu()]].to(object_list_embed.dtype)
                    else:
                        object_list_embed[4 + 3*nn::(3*self.knn+2), :] = embed_obj[torch.tensor(assigned_ids)[topk_indices[:,nn+NEIGHBOR_SHIFT].cpu()]].to(object_list_embed.dtype)
            else:


                object_list_embed[0::(3*self.knn+2), :] = selected_objid_embeds.to(object_list_embed.dtype)
                object_list_embed[1::(3*self.knn+2), :] = embed_img[assigned_ids]
                for nn in range(self.knn):
                    object_list_embed[2 + 3*nn::(3*self.knn+2), :] = embed_obj[assigned_ids]
                    if len(edges_assigned_ids) > 0 and max(edges_assigned_ids) < proj_edge_embed.shape[0]:
                        proj_edge_embed_s = proj_edge_embed[edges_assigned_ids]
                        object_list_embed[3 + 3*nn::(3*self.knn+2), :] = proj_edge_embed_s[nn::self.knn, :].to(object_list_embed.dtype)

                    object_list_embed[4 + 3*nn::(3*self.knn+2), :] = embed_obj[assigned_ids].to(object_list_embed.dtype)
        else:
            object_list_embed = torch.zeros((selected_objid_embeds.shape[0] * 3, selected_objid_embeds.shape[1]), dtype=selected_objid_embeds.dtype, device=selected_objid_embeds.device)
            object_list_embed[0::3, :] = selected_objid_embeds
            object_list_embed[1::3, :] = embed_obj[assigned_ids]
            object_list_embed[2::3, :] = embed_img[assigned_ids]
        return object_list_embed

    def get_min_max_coord(self, xyz, scene_mask):
        scene_mask = scene_mask.unsqueeze(-1).expand_as(xyz)
        masked_xyz_min = torch.where(scene_mask, xyz, torch.full_like(xyz, float('inf')))
        masked_xyz_max = torch.where(scene_mask, xyz, torch.full_like(xyz, float('-inf')))
        mins = masked_xyz_min.min(dim=1)[0]
        maxs = masked_xyz_max.max(dim=1)[0]
        return mins, maxs

    def _metric_quotient_tokens_for_sample(
        self,
        *,
        projected_objects,
        scene_radius,
        radius_valid,
        scene_mask,
        assigned_ids,
        question,
    ):
        if self.metric_quotient_token is None:
            return None
        if not self._metric_quotient_host_finalized:
            raise RuntimeError("metric quotient token was not finalized after host load")
        if scene_radius is None or radius_valid is None:
            raise ValueError("metric quotient token requires scene_radius and radius_valid")
        try:
            from .metric_quotient_token import intervene_scene_radii
        except ImportError as exc:
            raise ImportError(
                "metric quotient token requires the repository root on PYTHONPATH"
            ) from exc
        intervened = intervene_scene_radii(
            scene_radius,
            radius_valid,
            scene_mask,
            assigned_ids,
            question,
            intervention=self._metric_quotient_intervention,
            constant_radius_m=self._metric_quotient_constant_radius_m,
        )
        tokens, valid = self.metric_quotient_token(intervened, radius_valid)
        active = torch.as_tensor(scene_mask, device=valid.device).reshape(-1).to(torch.bool)
        if active.shape != valid.shape or torch.any(active & ~valid):
            raise RuntimeError("metric quotient token did not cover every active object")
        if tokens.shape[0] != projected_objects.shape[0]:
            raise RuntimeError("metric quotient token row count differs from object rows")
        return tokens.to(
            device=projected_objects.device, dtype=projected_objects.dtype
        )

    def forward_train(self, scene_feat, scene_img_feat, scene_locs, scene_mask, obj_ids, assigned_ids, scene_gnn_feats, foreground_ids, questions, answers, is_eval=False, **kwargs):


        scene_img_feat, scene_gnn_feats = self._apply_host_mode(
            scene_img_feat, scene_gnn_feats, scene_mask
        )
        object_embed, object_img_embed = self.encode_object_feat(
            scene_feat,
            scene_img_feat,
            scene_locs,
            object_radius=kwargs.get("scene_radius"),
            radius_valid=kwargs.get("radius_valid"),
            scene_mask=scene_mask,
        )
        object_embed, object_img_embed, scene_gnn_feats, scene_locs = self._cast_scene_inputs(
            object_embed, object_img_embed, scene_gnn_feats, scene_locs
        )


        device = object_embed.device
        batch_size = object_embed.shape[0]
        proj_object_embed = self.object_proj(object_embed)
        proj_object_img_embed = (
            self.object_img_proj(object_img_embed) if self.add_img_token else None
        )


        proj_edge_embed = None
        if self.host_mode == "full":
            # Kept raw until KNN has selected source/target identities.
            proj_edge_embed = scene_gnn_feats
        elif self.knn > 0 and not self.bbox_embed:
            scene_gnn_feats = torch.nan_to_num(torch.nn.functional.normalize(scene_gnn_feats, dim=-1))
            proj_edge_embed = self.edge_proj(scene_gnn_feats)


        if self.add_pos_emb:
            mins, maxs = self.get_min_max_coord(scene_locs[:, :, :3], scene_mask)
            pos_embed = self.pos_embedding(scene_locs[:, :, :3], input_range=[mins, maxs]) / 10
            if self.host_mode == "full":
                pos_embed = pos_embed.to(dtype=self._module_dtype(self.pos_proj))
            proj_pos_embed = self.pos_proj(pos_embed)
            proj_object_embed = proj_object_embed + proj_pos_embed
            if proj_object_img_embed is not None:
                proj_object_img_embed = proj_object_img_embed + proj_pos_embed

        proj_object_img_embed = self._zero_declared_image_slots(proj_object_img_embed, scene_img_feat)
        proj_scene_embed = None
        if self.add_scene_token:


            obj_embed = self.scene_init_proj(object_embed)


            scene_embed = obj_embed
            scene_embed = self.relation_module(scene_embed, src_key_padding_mask=~scene_mask)
            proj_scene_embed = self.scene_proj(scene_embed)

        input_embed_list, attn_list, target_list = [], [], []
        max_seq_len = 0

        p_0_embed = self.p_0_embed.to(device)
        p_1_embed = self.p_1_embed.to(device)
        object_list_intervals = []

        for i, question in enumerate(questions):

            prompt = f"{question} {self.role[1]}: "
            prompt_embed = self.get_text_emb(prompt, device=device).squeeze(0)
            metric_token_embed = self._metric_quotient_tokens_for_sample(
                projected_objects=proj_object_embed[i],
                scene_radius=(
                    kwargs.get("scene_radius")[i]
                    if kwargs.get("scene_radius") is not None
                    else None
                ),
                radius_valid=(
                    kwargs.get("radius_valid")[i]
                    if kwargs.get("radius_valid") is not None
                    else None
                ),
                scene_mask=scene_mask[i],
                assigned_ids=assigned_ids[i],
                question=question,
            )
            object_list_embed = self.get_object_list_embed(
                proj_object_embed[i],
                proj_object_img_embed[i] if self.add_img_token else None,
                proj_scene_embed[i] if self.add_scene_token else None,
                scene_mask[i],
                obj_ids[i],
                assigned_ids[i],
                proj_edge_embed[i] if proj_edge_embed is not None else None,
                scene_locs[i],
                foreground_ids[i],
                scene_feat[i],
                metric_token_embed,
            )


            object_list_intervals.append((p_0_embed.shape[0], p_0_embed.shape[0] + object_list_embed.shape[0]))
            wrapped_embed = torch.cat([p_0_embed, object_list_embed, p_1_embed, prompt_embed], dim=0)

            wrapped_attn = torch.ones(wrapped_embed.size()[:-1], dtype=torch.long).to(wrapped_embed.device)
            empty_target = (
                torch.ones(wrapped_attn.shape[0], dtype=torch.long).to(device).fill_(-100)
            )

            answer = answers[i] + self.end_sym
            to_regress_token = self.llama_tokenizer(answer, return_tensors="pt", add_special_tokens=False).to(device)

            answer_target = to_regress_token.input_ids.masked_fill(
                to_regress_token.input_ids == self.llama_tokenizer.pad_token_id, -100
            ).squeeze(0)

            to_regress_embed = self.get_text_emb(answer, device=device).squeeze(0)

            target = torch.cat([empty_target, answer_target], dim=0)
            input_embed = torch.cat([wrapped_embed, to_regress_embed], dim=0)
            attn = torch.cat([wrapped_attn, to_regress_token.attention_mask[0]], dim=0)
            input_embed_list.append(input_embed)
            attn_list.append(attn)
            target_list.append(target)
            max_seq_len = max(max_seq_len, target.shape[0])


        max_seq_len = min(2502, max_seq_len)

        def pad_and_trim(tensor_list, max_len, batch_first=True, padding_value=0):
            padded = pad_sequence(tensor_list, batch_first=batch_first, padding_value=padding_value)
            if padded.shape[1] > max_len:
                print("padded_sequence")
                return padded[:, :max_len]
            return padded

        input_embeds = pad_and_trim(input_embed_list, max_seq_len, batch_first=True, padding_value=0).to(device)
        targets = pad_and_trim(target_list, max_seq_len, batch_first=True, padding_value=-100).to(device)
        attention_mask = pad_and_trim(attn_list, max_seq_len, batch_first=True, padding_value=0).to(device)
        if self.bidirection:
            input_dtype = input_embeds.dtype
            causal_mask = torch.ones((max_seq_len, max_seq_len), dtype=input_dtype, device=device)
            causal_mask = torch.tril(causal_mask, diagonal=0)
            causal_mask = causal_mask[None, None, :, :].expand(input_embeds.shape[0], 1, -1, -1).clone()
            padding_mask = causal_mask[..., :].eq(1.0) * attention_mask[:, None, None, :].eq(0.0)
            causal_mask[..., :] = causal_mask[..., :].masked_fill(padding_mask, 0.0)
            for i in range(causal_mask.shape[0]):
                st, ed = object_list_intervals[i]
                causal_mask[i, :, st:ed, st:ed] = 1.0
            attention_mask = causal_mask


        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,

            )

        return dict(
            loss=outputs.loss,
            obj_norm=proj_object_embed.norm(dim=-1).mean().detach().cpu(),
            obj_img_norm=(
                proj_object_img_embed.norm(dim=-1).mean().detach().cpu()
                if proj_object_img_embed is not None
                else 0.0
            ),
            objid_norm=self.get_objid_embeds().norm(dim=-1).mean().detach().cpu(),
            scene_norm=proj_scene_embed.norm(dim=-1).mean().detach().cpu() if proj_scene_embed is not None else 0.,
            max_seq_len=max_seq_len
        )

    def evaluate(self, scene_feat, scene_img_feat, scene_locs, scene_mask, custom_prompt, obj_ids, assigned_ids, scene_gnn_feats, foreground_ids, is_eval=True, **kwargs):
        if self.low_resource:
            self.llama_model.config.use_cache = False
            gen_cfg = getattr(self.llama_model, "generation_config", None)
            if gen_cfg is not None:
                gen_cfg.use_cache = False
        scene_img_feat, scene_gnn_feats = self._apply_host_mode(
            scene_img_feat, scene_gnn_feats, scene_mask
        )
        object_embed, object_img_embed = self.encode_object_feat(
            scene_feat,
            scene_img_feat,
            scene_locs,
            object_radius=kwargs.get("scene_radius"),
            radius_valid=kwargs.get("radius_valid"),
            scene_mask=scene_mask,
        )
        object_embed, object_img_embed, scene_gnn_feats, scene_locs = self._cast_scene_inputs(
            object_embed, object_img_embed, scene_gnn_feats, scene_locs
        )
        device = object_embed.device
        batch_size = object_embed.shape[0]
        proj_object_embed = self.object_proj(object_embed)
        proj_object_img_embed = (
            self.object_img_proj(object_img_embed) if self.add_img_token else None
        )

        proj_edge_embed = None
        if self.host_mode == "full":
            proj_edge_embed = scene_gnn_feats
        elif self.knn > 0 and not self.bbox_embed:
            scene_gnn_feats = torch.nn.functional.normalize(scene_gnn_feats, dim=-1)
            proj_edge_embed = self.edge_proj(scene_gnn_feats)

        if self.add_pos_emb:
            mins, maxs = self.get_min_max_coord(scene_locs[:, :, :3], scene_mask)
            pos_embed = self.pos_embedding(scene_locs[:, :, :3], input_range=[mins, maxs]) / 10
            if self.host_mode == "full":
                pos_embed = pos_embed.to(dtype=self._module_dtype(self.pos_proj))
            proj_pos_embed = self.pos_proj(pos_embed)
            proj_object_embed = proj_object_embed + proj_pos_embed
            if proj_object_img_embed is not None:
                proj_object_img_embed = proj_object_img_embed + proj_pos_embed

        proj_object_img_embed = self._zero_declared_image_slots(proj_object_img_embed, scene_img_feat)
        proj_scene_embed = None
        if self.add_scene_token:


            obj_embed = self.scene_init_proj(object_embed)


            scene_embed = obj_embed
            scene_embed = self.relation_module(scene_embed, src_key_padding_mask=~scene_mask)
            proj_scene_embed = self.scene_proj(scene_embed)

        output_texts = []
        p_0_embed = self.p_0_embed.to(device).unsqueeze(0)
        p_1_embed = self.p_1_embed.to(device).unsqueeze(0)
        for i in range(batch_size):


            tmp_prompt = f" {custom_prompt[i]} {self.role[1]}: "
            prompt_embed = self.get_text_emb(tmp_prompt, device=device)
            metric_token_embed = self._metric_quotient_tokens_for_sample(
                projected_objects=proj_object_embed[i],
                scene_radius=(
                    kwargs.get("scene_radius")[i]
                    if kwargs.get("scene_radius") is not None
                    else None
                ),
                radius_valid=(
                    kwargs.get("radius_valid")[i]
                    if kwargs.get("radius_valid") is not None
                    else None
                ),
                scene_mask=scene_mask[i],
                assigned_ids=assigned_ids[i],
                question=custom_prompt[i],
            )
            object_list_embed = self.get_object_list_embed(
                proj_object_embed[i],
                proj_object_img_embed[i] if self.add_img_token else None,
                proj_scene_embed[i] if self.add_scene_token else None,
                scene_mask[i],
                obj_ids[i],
                assigned_ids[i],
                proj_edge_embed[i] if proj_edge_embed is not None else None,
                scene_locs[i],
                foreground_ids[i],
                scene_feat[i],
                metric_token_embed,
            )
            object_list_embed = object_list_embed.unsqueeze(0)
            wrapped_embed = torch.cat([p_0_embed, object_list_embed, p_1_embed, prompt_embed], dim=1)
            prefix_attention_mask = _generation_prefix_attention_mask(wrapped_embed)
            customized_attention_mask = None
            if self.bidirection:
                seq_len = wrapped_embed.shape[1]
                customized_attention_mask = torch.ones((seq_len, seq_len), dtype=wrapped_embed.dtype, device=device)
                customized_attention_mask = torch.tril(customized_attention_mask, diagonal=0)
                customized_attention_mask = customized_attention_mask[None, None, :, :].expand(1, 1, -1, -1).clone()
                st, ed = p_0_embed.shape[1], p_0_embed.shape[1] + object_list_embed.shape[1]
                customized_attention_mask[:, :, st:ed, st:ed] = 1.0


            stop_words_ids = [torch.tensor([ 14711]).to(wrapped_embed.device), torch.tensor([198,  14711]).to(wrapped_embed.device), torch.tensor([82, 29]).to(wrapped_embed.device), torch.tensor([524]).to(wrapped_embed.device)]

            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
            num_beams = 1 if self.low_resource else 5
            with self.maybe_autocast():
                outputs = self.llama_model.generate(
                    inputs_embeds=wrapped_embed,
                    attention_mask=prefix_attention_mask,
                    max_new_tokens=self.max_txt_len,

                    stopping_criteria=stopping_criteria,
                    use_cache=not self.low_resource,
                    num_beams=num_beams,

                    min_length=1,

                    repetition_penalty=3.0,
                    length_penalty=1,
                    temperature=1.0,
                    customized_mask=customized_attention_mask,
                    pad_token_id=self.llama_tokenizer.eos_token_id
                )
            output_token = outputs[0]
            output_text = self.llama_tokenizer.decode(output_token)
            output_text = output_text.split(self.end_sym)[0]
            output_text = output_text.replace('  ', ' ').replace(' .', '.').strip()
            output_text = recover_caption(output_text, assigned_ids[i].tolist())
            output_texts.append(output_text)
        return output_texts

    def forward(self, **kwargs):
        if "answers" in kwargs:
            return self.forward_train(**kwargs)
        if "custom_prompt" in kwargs:
            return self.evaluate(**kwargs)
        return None

    def _get_text_len(self, text):
        return self.llama_tokenizer(text, return_tensors="pt").input_ids.shape[1]

    def maybe_autocast(self, dtype=None):


        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype or preferred_cuda_dtype())
        else:
            return contextlib.nullcontext()

    @property
    def device(self):
        return list(self.parameters())[0].device
