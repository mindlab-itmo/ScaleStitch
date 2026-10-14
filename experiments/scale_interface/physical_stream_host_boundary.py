"""Measurement insertion into causal object-only and full image/graph hosts."""
import torch
from .role_bound_metric_schema import canonical_parts


def encode_packet(core, records, queried_ids, *, representation, codec=None):
    if representation == 'absent':
        return None, []
    if representation not in ('lexical', 'trim', 'codec'):
        raise ValueError('unsupported physical representation')
    parts = canonical_parts(records, queried_ids)
    encode = lambda text: core.llama_tokenizer(text, add_special_tokens=False)['input_ids']
    ids = [encode(part) for part in parts]
    if sum(ids, []) != encode(''.join(parts)):
        raise ValueError('packet segmentation changes native tokenizer IDs')
    device = next(core.llama_model.parameters()).device
    blocks = [core.get_text_emb(part, device=device).squeeze(0).detach() for part in parts]
    for index in (1, 3):
        if representation == 'codec':
            if codec is None:
                raise ValueError('codec representation requires explicit weights')
            blocks[index] = codec(blocks[index])
        elif representation == 'trim':
            if len(ids[index]) < 2 or core.llama_tokenizer.decode([ids[index][0]]) != ' ':
                raise ValueError('lossless trim requires a standalone leading blank')
            blocks[index] = blocks[index][1:]
    return torch.cat(blocks), ids


def generation_question_boundary(question, role, encode):
    prefix, suffix = f' {question}', f' {role}: '
    left, right = list(encode(prefix)), list(encode(suffix))
    if left + right != list(encode(prefix + suffix)):
        raise ValueError('generation boundary changes the native prompt tokenization')
    return len(left)


def splice_inputs(kwargs, payload, index, *, maximum_length=2502):
    if payload is None:
        return kwargs, 0
    x, mask = kwargs['inputs_embeds'], kwargs['attention_mask']
    labels = kwargs.get('labels')
    if (x.ndim != 3 or x.shape[0] != 1 or mask.ndim != 2 or mask.shape != x.shape[:2]
            or not torch.all(mask == 1) or kwargs.get('customized_mask') is not None):
        raise ValueError('requires one unpadded, ordinary causal prefix')
    if (payload.ndim != 2 or payload.shape[1] != x.shape[-1] or not torch.isfinite(payload).all()
            or not isinstance(index, int) or isinstance(index, bool) or not 0 <= index <= x.shape[1]
            or x.shape[1] + len(payload) > maximum_length):
        raise ValueError('invalid packet, insertion point or sequence budget')
    if labels is not None:
        if labels.shape != mask.shape:
            raise ValueError('target shape differs from native input')
        targets = torch.where(labels[0] != -100)[0]
        if not len(targets) or index > int(targets[0]):
            raise ValueError('packet must precede every supervised answer token')
    if len(payload) == 0:
        return kwargs, 0
    packet = payload.to(device=x.device, dtype=x.dtype).unsqueeze(0)
    count = len(payload)
    new = {**kwargs, 'inputs_embeds':torch.cat((x[:,:index],packet,x[:,index:]),1),
           'attention_mask':torch.cat((mask[:,:index],mask.new_ones((1,count)),mask[:,index:]),1)}
    if labels is not None:
        new['labels'] = torch.cat((labels[:,:index],labels.new_full((1,count),-100),labels[:,index:]),1)
    fields = ['inputs_embeds','attention_mask'] + (['labels'] if labels is not None else [])
    for field in fields:
        recovered = torch.cat((new[field][:,:index],new[field][:,index+count:]),1)
        if not torch.equal(recovered,kwargs[field]):
            raise RuntimeError('native input/target parity failed')
    return new, len(fields)


class HostPhysicalStream:
    def __init__(self, core, *, generation_options=None):
        if core.bidirection or core.host_mode not in ('object_only', 'full'):
            raise ValueError('requires a causal object-only or full host')
        if core.host_mode == 'object_only' and core.knn != 0:
            raise ValueError('object-only boundary requires knn=0')
        if core.host_mode == 'full' and (
            core.knn < 1 or not core.add_img_token or core.add_scene_token or core.bbox_embed
        ):
            raise ValueError('full boundary requires image and graph tokens without scene/bbox tokens')
        self.core = core
        self.generation_options = dict(generation_options or {})
        if set(self.generation_options) - {'max_new_tokens','num_beams','do_sample','use_cache'}:
            raise ValueError('undeclared generation override')
        self.active = False
        self.armed = False

    def __enter__(self):
        core = self.core
        if self.active or getattr(core,'_rps_boundary_active',False):
            raise RuntimeError('nested/concurrent boundary not supported')
        core._rps_boundary_active = True
        self.original_assembler = core.get_object_list_embed
        self.original_provider = core._metric_quotient_tokens_for_sample
        self.original_generate = core.llama_model.generate
        core.get_object_list_embed = self._assemble
        core._metric_quotient_tokens_for_sample = lambda **kwargs: None
        core.llama_model.generate = self._generate
        self.pre = core.llama_model.register_forward_pre_hook(self._forward, with_kwargs=True)
        self.active = True
        return self

    def arm(self, payload, offset, *, mode, placement='question_tail'):
        if not self.active or self.armed or mode not in ('train','generate') or not isinstance(offset,int) or offset < 0:
            raise ValueError('one complete armed request is required')
        if placement not in ('question_tail', 'object_end'):
            raise ValueError('unknown measurement placement')
        self.payload, self.offset, self.mode = payload, offset, mode
        self.placement = placement
        self.audit = {'assembler_calls':0,'boundary_calls':0,'native_parity_assertions':0}
        self.armed = True

    def _assemble(self, *args, **kwargs):
        if not self.armed:
            raise RuntimeError('unarmed host call')
        result = self.original_assembler(*args, **kwargs)
        self.audit['assembler_calls'] += 1
        self.index = len(self.core.p_0_embed) + len(result)
        if self.placement == 'question_tail':
            self.index += len(self.core.p_1_embed) + self.offset
        return result

    def _apply(self, kwargs):
        if not self.armed or self.audit['assembler_calls'] != 1 or self.audit['boundary_calls']:
            raise RuntimeError('missing/repeated assembly or repeated packet insertion')
        result, count = splice_inputs(kwargs,self.payload,self.index)
        self.audit.update(boundary_calls=1,native_parity_assertions=count,
                          native_tokens=kwargs['inputs_embeds'].shape[1],
                          packet_tokens=0 if self.payload is None else len(self.payload),
                          total_tokens=result['inputs_embeds'].shape[1])
        return result

    def _forward(self, model, args, kwargs):
        if self.armed and self.mode == 'generate':
            return args, kwargs
        if args or kwargs.get('labels') is None:
            raise ValueError('training boundary requires keyword inputs and labels')
        return args, self._apply(kwargs)

    def _generate(self, *args, **kwargs):
        if args or not self.armed or self.mode != 'generate':
            raise ValueError('generation requires one explicitly armed keyword request')
        return self.original_generate(**self._apply({**kwargs,**self.generation_options}))

    def finish(self):
        if not self.armed or self.audit['assembler_calls'] != 1 or self.audit['boundary_calls'] != 1:
            raise RuntimeError('incomplete boundary request')
        self.armed = False
        return dict(self.audit)

    def __exit__(self, *exc):
        self.pre.remove()
        self.core.get_object_list_embed = self.original_assembler
        self.core._metric_quotient_tokens_for_sample = self.original_provider
        self.core.llama_model.generate = self.original_generate
        del self.core._rps_boundary_active
        self.active = self.armed = False
