from __future__ import annotations

import torch
from tqdm import tqdm

from sampling import sample
from .validate import _rand_timesteps
from losses import unified_loss, deep_supervised_loss
# AMP autocast compatibility shim: prefer torch.amp API
try:
    from torch.amp import autocast as _amp_autocast  # type: ignore
    def autocast(enabled: bool = False):
        return _amp_autocast('cuda', enabled=enabled)
except Exception:
    try:
        from torch.cuda.amp import autocast as _cuda_autocast  # type: ignore
        def autocast(enabled: bool = False):
            return _cuda_autocast(enabled=enabled)
    except Exception:
        def autocast(enabled: bool = False):
            from contextlib import contextmanager
            @contextmanager
            def _noop():
                yield
            return _noop()


def train_one_epoch(trainer, epoch_idx: int):
    """Train for one epoch using the provided UnifiedTrainer instance.

    Accumulates epoch-average losses and handles periodic sampling.
    """
    cfg = trainer.cfg
    model, ddpm, device = trainer.model, trainer.ddpm, trainer.device
    channels, img_size = trainer.channels, trainer.img_size

    model.train()
    pbar = tqdm(trainer.train_loader, desc=f"[{cfg.mode}|{trainer.ds_key}] epoch {epoch_idx+1}/{cfg.epochs}")

    eb = 0
    e_loss = e_x0 = e_cons = e_tot = 0.0

    for imgs, _ in pbar:
        imgs = imgs.to(device)
        t = _rand_timesteps(ddpm, imgs, device)

        with autocast(enabled=getattr(trainer, 'use_amp', False)):
            if cfg.mode == "dsd":
                total, parts = deep_supervised_loss(model, ddpm, imgs, t, cfg.dsd_w_aux_eps, cfg.dsd_w_aux_x0)
            else:
                total, parts = unified_loss(model, ddpm, imgs, t, multi_task=(cfg.mode=="multi"),
                                            w_x0=cfg.w_x0, w_consistency=cfg.w_consistency, multi_variant=cfg.multi_variant)

        trainer.optim.zero_grad()
        if getattr(trainer, 'use_amp', False) and getattr(trainer, 'scaler', None) is not None:
            trainer.scaler.scale(total).backward()
            trainer.scaler.step(trainer.optim)
            trainer.scaler.update()
        else:
            total.backward()
            trainer.optim.step()

        trainer.step += 1

        eb += 1
        e_loss += float(parts["loss"]) 
        if cfg.mode in ("multi", "dsd"):
            e_x0  += float(parts.get("loss_x0", 0.0))
            e_cons+= float(parts.get("loss_cons", 0.0))
            e_tot += float(parts.get("loss_total", 0.0))

        # For single-task (and other non-multi modes), keep step-based grids.
        # Multi-task grids are saved at end of each epoch (see training.run).
        if (trainer.step % cfg.sample_every) == 0 and cfg.mode != "multi":
            try:
                with torch.no_grad():
                    grid_path = trainer.grid_dir / f"{trainer.file_prefix}_samples_step{trainer.step}.png"
                    sample(model, ddpm,
                           shape=(cfg.n_sample, channels, img_size, img_size),
                           device=device, save_path=str(grid_path))
            except Exception as e:
                print(f"[warn|sample-grid|{cfg.mode}|{trainer.ds_key}] step={trainer.step} failed: {e}")

    denom = max(1, eb)
    avg = {
        'loss': e_loss/denom,
        'loss_x0': (e_x0/denom) if cfg.mode in ("multi", "dsd") else None,
        'loss_cons': (e_cons/denom) if cfg.mode in ("multi", "dsd") else None,
        'loss_total': (e_tot/denom) if cfg.mode in ("multi", "dsd") else None,
    }
    if cfg.mode in ("multi", "dsd"):
        print(f"[train] epoch {epoch_idx+1}: loss={avg['loss']:.6f} | loss_x0={avg['loss_x0']:.6f} | loss_cons={avg['loss_cons']:.6f} | loss_total={avg['loss_total']:.6f}")
    else:
        print(f"[train] epoch {epoch_idx+1}: loss={avg['loss']:.6f}")

    return avg
