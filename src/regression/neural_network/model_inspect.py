import torch
import math


def check_params_and_grads(model, topk=5):
    bad_params = []
    bad_grads = []

    with torch.no_grad():
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue

            # # Parameter values
            if not torch.isfinite(p).all():
                bad_params.append(name)

            # # Gradients (may be None before backward)
            if p.grad is not None and (not torch.isfinite(p.grad).all()):
                bad_grads.append(name)

    return bad_params, bad_grads



@torch.no_grad()
def grad_param_norms(model):
    stats = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        p_norm = p.norm().item()
        g_norm = p.grad.norm().item() if p.grad is not None else float("nan")
        stats[name] = {"param_norm": p_norm, "grad_norm": g_norm}
    return stats