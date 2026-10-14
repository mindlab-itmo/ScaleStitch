"""Load causal object-only/full hosts and measured feature or token adapters."""
from types import SimpleNamespace
import random

import numpy as np
import torch

from .deployment import ScaleStitchDeployment
from .metric_quotient_token import AnchoredQuotientToken


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def namespace(value):
    return SimpleNamespace(**{k: namespace(v) if isinstance(v, dict) else v
                              for k, v in value.items()})


def load_host(config, *, device="cuda"):
    from .host import Graph3DLLM

    model_config = dict(config["model"])
    feature_mode = config.get("feature_mode", "deployment")
    if feature_mode not in ("deployment", "native"):
        raise ValueError("host.feature_mode must be deployment or native")
    if feature_mode == "native":
        if any(type(model_config.get(key)) is not int or model_config[key] != 1024
               for key in ("source_feature_dim", "input_dim")):
            raise ValueError("native features require explicit source_feature_dim=input_dim=1024")
        if "deployment" in config or "aqt" in config:
            raise ValueError("native features reject deployment and aqt fields; use an external measurement packet")
    if model_config["host_mode"] not in ("object_only", "full") or model_config["bidirection"]:
        raise ValueError("these experiments require a causal object_only or full host")
    if model_config["host_mode"] == "full":
        if (model_config.get("add_img_token") is not True or model_config.get("img_input_dim") != 1024
                or model_config.get("gt_pretrain") is not True
                or model_config.get("add_scene_token") is not False
                or model_config.get("bbox_embed") is not False
                or type(model_config.get("knn")) is not int or model_config["knn"] < 1
                or type(model_config.get("max_knn")) is not int
                or model_config["max_knn"] < model_config["knn"]):
            raise ValueError("full recovery requires GT objects, real images, positive KNN and no scene/bbox tokens")
        if "aqt" in config:
            raise ValueError("full recovery uses an external measurement packet, not per-object quotient tokens")
    core = Graph3DLLM(namespace({"debug": False, "device": device,
                                "model": model_config, "lora": config["lora"]}))
    payload = torch.load(config["host_checkpoint"], map_location="cpu", weights_only=True)
    state = payload.get("model", payload)
    expected = core.state_dict()
    unexpected = set(state) - set(expected)
    if unexpected:
        raise ValueError(f"unexpected host tensors: {sorted(unexpected)}")
    required = {name for name, p in core.named_parameters() if p.requires_grad}
    # The standalone encoder-layer template is unused in the object-only host.
    required = {name for name in required if not name.startswith("encoder_layer.")}
    required |= {name for name in expected if name.startswith("object_proj.")}
    if model_config["host_mode"] == "full":
        prefixes = ("object_img_proj.", "edge_proj.")
        if model_config.get("add_pos_emb"):
            prefixes += ("pos_proj.",)
        required |= {name for name in expected if name.startswith(prefixes)}
    missing = required - state.keys()
    if missing:
        raise ValueError(f"missing host tensors: {sorted(missing)}")
    for name, value in state.items():
        if not isinstance(value, torch.Tensor) or value.shape != expected[name].shape:
            raise ValueError(f"host tensor shape mismatch: {name}")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"nonfinite host tensor: {name}")
    core.load_state_dict(state, strict=False)
    del state, payload, expected
    if feature_mode == "deployment":
        bundle = torch.load(config["deployment"], map_location="cpu", weights_only=True)
        core.scalestitch = ScaleStitchDeployment(bundle, execution_mode="inference")
        core._scalestitch_deployment_bundle = True
    # Native 1024-wide inputs retain the host's original L2 normalization and
    # checkpoint-loaded object projection. No learned bridge is constructed.
    core._scalestitch_freeze_host = True
    core.to(device).requires_grad_(False).eval()
    # Fixed prompt embeddings are intentionally retained from host construction.
    # This matches the executed host's checkpoint-loading order.
    core.p_0_embed = core.p_0_embed.to(device)
    core.p_1_embed = core.p_1_embed.to(device)
    if config.get("aqt"):
        attach_aqt(core, bundle["radius_contract"], **config["aqt"])
    return core


def attach_aqt(core, radius, *, checkpoint=None, variant="anchored_quotient_token",
               intervention="true", constant_radius_m=1.0):
    def embed(text):
        ids = core.llama_tokenizer(text, add_special_tokens=False)["input_ids"]
        return core.llama_embed_tokens(torch.tensor(ids, device=core._llama_embed_device())).detach().float()

    module = AnchoredQuotientToken(
        core.p_0_embed.shape[-1], radius["log_radius_mean"], radius["log_radius_std"],
        center_embedding=embed(" 65")[-1],
        direction_initialization=(embed(" 145")[-1] - embed(" 29")[-1]) / 2.0,
        prefix_embeddings=embed(" radius"), suffix_embeddings=embed(" cm"), variant=variant)
    if checkpoint:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = payload.get("state", payload)
        direction = state["direction"]
        if direction.shape != module.direction.shape or not torch.isfinite(direction).all():
            raise ValueError("invalid quotient-token direction")
        with torch.no_grad():
            module.direction.copy_(direction)
    core.metric_quotient_token = module.to(core._llama_embed_device()).requires_grad_(False).eval()
    core._metric_quotient_intervention = intervention
    core._metric_quotient_constant_radius_m = constant_radius_m
    core._metric_quotient_execution_mode = "inference"
    core._metric_quotient_host_finalized = True
    return module
