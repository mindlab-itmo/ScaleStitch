"""Restricted first-token decision for two native object identifiers."""
import re

import torch


def native_candidate_tokens(tokenizer, candidate_identifiers):
    names = tuple(candidate_identifiers)
    if (len(names) != 2 or len(set(names)) != 2
            or any(not isinstance(name, str) or re.fullmatch(r"<OBJ[0-9]{3}>", name) is None
                   for name in names)):
        raise ValueError("exactly two distinct native candidate identifiers are required")
    encoded = [tokenizer(name, add_special_tokens=False)["input_ids"] for name in names]
    if (any(len(ids) != 1 or isinstance(ids[0], bool) or not isinstance(ids[0], int)
            or ids[0] < 0 for ids in encoded) or encoded[0] == encoded[1]):
        raise ValueError("candidate identifiers must be distinct single native tokens")
    return names, [ids[0] for ids in encoded]


def candidate_token_decision(model, tokenizer, *, inputs_embeds, attention_mask,
                             candidate_identifiers):
    """Compare two eligible logits without changing the captured scene prefix.

    This endpoint requires each native identifier to be one distinct token.
    It measures a restricted next-token decision, not free-answer generation.
    """
    names, token_ids = native_candidate_tokens(tokenizer, candidate_identifiers)
    if (inputs_embeds.ndim != 3 or inputs_embeds.shape[0] != 1
            or attention_mask.shape != inputs_embeds.shape[:2]):
        raise ValueError("one complete host prefix is required")
    with torch.no_grad():
        result = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                       use_cache=False, return_dict=True, logits_to_keep=1)
        logits = result.logits
        if logits.ndim != 3 or logits.shape[:2] != (1, 1) or max(token_ids) >= logits.shape[-1]:
            raise ValueError("unexpected native next-token vocabulary or output shape")
        values = logits[0, 0].float()
        if not torch.isfinite(values).all():
            raise ValueError("nonfinite next-token logits")
        eligible = values[token_ids]
        winner = int(eligible.argmax().item())
        info = dict(candidate_identifiers=list(names), candidate_token_ids=token_ids,
                    candidate_logits=eligible.cpu().tolist(),
                    candidate_conditional_probabilities=eligible.softmax(0).cpu().tolist(),
                    pair_probability=float(values.softmax(0)[token_ids].sum().item()),
                    unrestricted_first_token_id=int(values.argmax().item()),
                    score_space="raw model logits before generation processors",
                    tie=bool(eligible[0].item() == eligible[1].item()),
                    decision_rule="maximum first-token logit; exact ties use question order")
        chosen = torch.tensor([[token_ids[winner]]], device=inputs_embeds.device, dtype=torch.long)
    return chosen, info
