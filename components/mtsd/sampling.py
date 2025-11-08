from __future__ import annotations
import math
from pathlib import Path
import torch
from torchvision import utils as vutils
from typing import Tuple


@torch.no_grad()
def sample(model, ddpm, shape, device, save_path: str = "samples.png") -> Tuple[str, torch.Tensor]:
    model.eval()
    T = ddpm.cfg.timesteps
    x = torch.randn(shape, device=device)
    for i in reversed(range(T)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        beta_t = ddpm._extract(ddpm.betas, t, x.shape)
        sqrt_one_minus_ac = ddpm._extract(ddpm.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alpha = ddpm._extract(ddpm.sqrt_recip_alphas, t, x.shape)
        preds = model(x, t.float())
        pred_eps = preds["eps"] if isinstance(preds, dict) else preds
        x0_pred = (x - sqrt_one_minus_ac * pred_eps) * sqrt_recip_alpha
        alphas = ddpm._extract(ddpm.alphas, t, x.shape)
        alphas_cum = ddpm._extract(ddpm.alphas_cumprod, t, x.shape)
        alphas_cum_prev = ddpm._extract(ddpm.alphas_cumprod_prev, t, x.shape)
        posterior_var = ddpm._extract(ddpm.posterior_variance, t, x.shape)
        posterior_mean = (
            (beta_t * torch.sqrt(alphas_cum_prev) / (1.0 - alphas_cum)) * x0_pred
            + ((torch.sqrt(alphas) * (1.0 - alphas_cum_prev)) / (1.0 - alphas_cum)) * x
        )
        noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
        x = posterior_mean + torch.sqrt(posterior_var) * noise
    grid = vutils.make_grid((x.clamp(-1, 1) + 1) * 0.5, nrow=int(math.sqrt(shape[0])))
    vutils.save_image(grid, save_path)
    try:
        p = Path(save_path)
        assert p.exists() and p.stat().st_size > 0, f"Sample grid not saved: {save_path}"
    except Exception as e:
        raise AssertionError(f"Failed to save sample grid '{save_path}': {e}")
    return save_path, x.detach().cpu()
