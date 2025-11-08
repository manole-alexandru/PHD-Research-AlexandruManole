from __future__ import annotations

import torch
import torch.nn.functional as F


def _rand_timesteps(ddpm, imgs, device):
    return torch.randint(0, ddpm.cfg.timesteps, (imgs.size(0),), device=device)


@torch.no_grad()
def evaluate_mse(model, ddpm, data_loader, device, mode_multi: bool):
    model.eval()
    tot_eps, tot_x0, denom = 0.0, 0.0, 0
    for imgs, _ in data_loader:
        imgs = imgs.to(device)
        t = _rand_timesteps(ddpm, imgs, device)
        x_t, noise = ddpm.q_sample(imgs, t)
        preds = model(x_t, t.float())
        if mode_multi:
            eps = preds["eps"]; x0p = preds["x0"]
            mse_eps = F.mse_loss(eps, noise, reduction='sum')
            mse_x0  = F.mse_loss(x0p, imgs, reduction='sum')
            tot_eps += mse_eps.item()
            tot_x0  += mse_x0.item()
        else:
            eps = preds["eps"] if isinstance(preds, dict) else preds
            mse_eps = F.mse_loss(eps, noise, reduction='sum')
            tot_eps += mse_eps.item()
        denom += imgs.numel()
    denom = max(1, denom)
    out = {"loss": tot_eps / denom}
    out["loss_x0"] = (tot_x0 / denom) if mode_multi else None
    return out

