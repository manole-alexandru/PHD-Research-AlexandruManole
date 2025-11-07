from __future__ import annotations

import torch
import torch.nn.functional as F

from diffusion import DDPM


def unified_loss(model, ddpm: DDPM, x0, t, multi_task: bool,
                 w_x0: float = 1.0, w_consistency: float = 0.1,
                 multi_variant: str = "eps_x0_consistency"):
    x_t, noise = ddpm.q_sample(x0, t)
    preds = model(x_t, t.float())
    if not multi_task:
        loss_eps = F.mse_loss(preds, noise)
        return loss_eps, {"loss": loss_eps.detach()}

    pred_eps = preds["eps"]; pred_x0 = preds["x0"]
    loss_eps = F.mse_loss(pred_eps, noise)
    loss_x0  = F.mse_loss(pred_x0, x0)

    sqrt_ac = ddpm._extract(ddpm.sqrt_alphas_cumprod, t, x_t.shape)
    sqrt_om = ddpm._extract(ddpm.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
    x0_from_eps = (x_t - sqrt_om * pred_eps) / (sqrt_ac + 1e-8)
    eps_from_x0 = (x_t - sqrt_ac * pred_x0) / (sqrt_om + 1e-8)
    cons1 = F.mse_loss(x0_from_eps.detach(), pred_x0)
    cons2 = F.mse_loss(eps_from_x0.detach(), pred_eps)
    loss_cons = 0.5 * (cons1 + cons2)

    if multi_variant == "eps_x0_consistency":
        total = loss_eps + w_x0 * loss_x0 + w_consistency * loss_cons
    else:
        total = loss_eps + w_x0 * loss_x0 + w_consistency * loss_cons

    return total, {
        "loss": loss_eps.detach(),
        "loss_x0": loss_x0.detach(),
        "loss_cons": loss_cons.detach(),
        "loss_total": total.detach(),
    }


def deep_supervised_loss(model, ddpm: DDPM, x0, t,
                         w_aux_eps: float = 0.5, w_aux_x0: float = 0.5):
    x_t, noise = ddpm.q_sample(x0, t)
    eps_list = model.forward_with_heads(x_t, t.float())
    eps_main = eps_list[-1]
    loss_main = F.mse_loss(eps_main, noise)

    aux_eps = 0.0
    aux_x0 = 0.0
    sqrt_ac = ddpm._extract(ddpm.sqrt_alphas_cumprod, t, x_t.shape)
    sqrt_om = ddpm._extract(ddpm.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
    for ep in eps_list[:-1]:
        aux_eps = aux_eps + F.mse_loss(ep, noise)
        x0_from_ep = (x_t - sqrt_om * ep) / (sqrt_ac + 1e-8)
        aux_x0 = aux_x0 + F.mse_loss(x0_from_ep, x0)

    total = loss_main + w_aux_eps * aux_eps + w_aux_x0 * aux_x0
    return total, {
        "loss": loss_main.detach(),
        "loss_x0": torch.as_tensor(aux_x0).detach(),
        "loss_cons": torch.as_tensor(aux_eps).detach(),
        "loss_total": torch.as_tensor(total).detach(),
    }

