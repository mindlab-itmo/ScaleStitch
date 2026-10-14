"""Full-host channel controls that retain projected token shapes and positions."""
from contextlib import contextmanager

import torch


@contextmanager
def channel_control(core, mode="original"):
    if mode == "original":
        yield
        return
    if mode not in ("zero_images", "zero_relations") or getattr(core, "host_mode", None) != "full":
        raise ValueError("channel_intervention requires original or a declared full-host channel control")
    module = core.object_img_proj if mode == "zero_images" else core.edge_proj
    # Zero after the projection, including its bias; keep every token slot.
    handle = module.register_forward_hook(lambda module, args, output: torch.zeros_like(output))
    try:
        yield
    finally:
        handle.remove()
