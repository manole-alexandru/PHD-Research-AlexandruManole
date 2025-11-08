from __future__ import annotations
import math
from pathlib import Path
import torch
from torchvision import utils as vutils
from typing import Tuple, Literal


@torch.no_grad()
def sample(
    model,
    ddpm,
    shape,
    device,
    save_path: str = "samples.png",
    # Multi-task generation mode: 'eps' uses epsilon head, 'x0' uses x0 head,
    # 'combined' blends x0 estimates from both heads. Defaults keep backwards compatibility.
    multi_gen: Literal["eps", "x0", "combined"] = "eps",
    combine_weight: float = 0.5,
) -> Tuple[str, torch.Tensor]:
    model.eval()
    T = ddpm.cfg.timesteps
    x = torch.randn(shape, device=device)
    for i in reversed(range(T)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        beta_t = ddpm._extract(ddpm.betas, t, x.shape)
        sqrt_one_minus_ac = ddpm._extract(ddpm.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alpha = ddpm._extract(ddpm.sqrt_recip_alphas, t, x.shape)
        preds = model(x, t.float())
        if isinstance(preds, dict):
            # Multi-task case: both heads may be available
            has_eps = ("eps" in preds)
            has_x0 = ("x0" in preds)
            if multi_gen == "eps":
                if not has_eps:
                    raise ValueError("Requested multi_gen='eps' but model did not return 'eps' head")
                pred_eps = preds["eps"]
                x0_pred = (x - sqrt_one_minus_ac * pred_eps) * sqrt_recip_alpha
            elif multi_gen == "x0":
                if not has_x0:
                    raise ValueError("Requested multi_gen='x0' but model did not return 'x0' head")
                pred_x0 = preds["x0"]
                x0_pred = pred_x0
            elif multi_gen == "combined":
                if not (has_eps and has_x0):
                    raise ValueError("Requested multi_gen='combined' but model did not return both 'eps' and 'x0' heads")
                pred_eps = preds["eps"]
                pred_x0 = preds["x0"]
                # Convert eps to x0 estimate, then blend with direct x0 head
                x0_from_eps = (x - sqrt_one_minus_ac * pred_eps) * sqrt_recip_alpha
                w = float(combine_weight)
                w = max(0.0, min(1.0, w))
                x0_pred = (1.0 - w) * x0_from_eps + w * pred_x0
            else:
                raise ValueError(f"Unknown multi_gen mode: {multi_gen}")
        else:
            # Single-head models (single-task or DSD forward()): use eps
            pred_eps = preds
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
