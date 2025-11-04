from __future__ import annotations
from pathlib import Path
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models import TinyUNet, DeepSupervisedUNet
from diffusion import DDPM, DiffusionConfig
from data_utils import make_dataloader
from sampling import sample
from metrics_utils import (
    denorm, dump_images, compute_fid, save_curves_unified_prefixed
)


def unified_loss(model, ddpm: DDPM, x0, t, multi_task: bool, w_x0=1.0, w_consistency=0.1, multi_variant: str = "eps_x0_consistency"):
    x_t, noise = ddpm.q_sample(x0, t)
    preds = model(x_t, t.float())
    if not multi_task:
        loss_eps = F.mse_loss(preds, noise)
        return loss_eps, {"loss": loss_eps.detach()}
    # Multi-task variants
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

    if multi_variant == "eps_x0_consistency":  # current default variant name
        total = loss_eps + w_x0 * loss_x0 + w_consistency * loss_cons
    else:
        # fallback to default behavior if unknown variant
        total = loss_eps + w_x0 * loss_x0 + w_consistency * loss_cons

    return total, {
        "loss": loss_eps.detach(),
        "loss_x0": loss_x0.detach(),
        "loss_cons": loss_cons.detach(),
        "loss_total": total.detach()
    }


def deep_supervised_loss(model: DeepSupervisedUNet, ddpm: DDPM, x0, t,
                         w_aux_eps: float = 0.5, w_aux_x0: float = 0.5):
    x_t, noise = ddpm.q_sample(x0, t)
    eps_list = model.forward_with_heads(x_t, t.float())
    # Use final head as main
    eps_main = eps_list[-1]
    loss_main = F.mse_loss(eps_main, noise)

    # Auxiliary losses from intermediate heads
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
        "loss_x0": aux_x0.detach(),
        "loss_cons": aux_eps.detach(),
        "loss_total": total.detach(),
    }


def train_unified(
    save_root: str = "runs",
    mode: str = "single",
    data: str = "mnist",
    epochs: int = 1,
    batch_size: int = 128,
    lr: float = 2e-4,
    timesteps: int = 200,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    base: int = 32,
    time_dim: int = 128,
    n_sample: int = 64,
    sample_every: int = 500,
    val_split: float = 0.05,
    fid_eval_images: int = 1024,
    w_x0: float = 1.0,
    w_consistency: float = 0.1,
    experiment_dir: str | Path | None = None,
    # Multi-task variant selector (for mode=="multi")
    multi_variant: str = "eps_x0_consistency",
    # Deep Supervised Diffusion weights (for mode=="dsd")
    dsd_w_aux_eps: float = 0.5,
    dsd_w_aux_x0: float = 0.5,
):
    assert mode in ["single", "multi", "dsd"]
    ds_key = "cifar10" if data.lower() in ["cifar10", "cifar"] else "mnist"

    base_dir = Path(experiment_dir) if experiment_dir is not None else (Path(save_root) / "experiment")
    images_dir = base_dir / "images"
    metrics_dir = base_dir / "metrics"
    fid_dir = base_dir / "fid" / f"{ds_key}_{mode}"
    ckpt_dir = base_dir / "checkpoints"
    fid_train_real = fid_dir / "train_real"
    fid_train_fake = fid_dir / "train_fake"
    fid_val_real   = fid_dir / "val_real"
    fid_val_fake   = fid_dir / "val_fake"
    file_prefix = f"{ds_key}_{mode}"
    for d in [images_dir, metrics_dir, fid_train_real, fid_train_fake, fid_val_real, fid_val_fake, ckpt_dir]:
        d.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channels = 1 if ds_key == "mnist" else 3
    img_size = 28 if ds_key == "mnist" else 32

    train_loader, val_loader, train_fid_loader = make_dataloader(ds_key, batch_size, img_size, channels, val_split=val_split)

    if mode == "dsd":
        model = DeepSupervisedUNet(in_channels=channels, base=base, time_dim=time_dim).to(device)
    else:
        model = TinyUNet(in_channels=channels, base=base, time_dim=time_dim, multi_task=(mode=="multi")).to(device)
    ddpm = DDPM(DiffusionConfig(timesteps=timesteps, beta_start=beta_start, beta_end=beta_end)).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    train_steps = []
    tr_loss_main, tr_loss_x0, tr_loss_cons, tr_loss_total = [], [], [], []
    val_epochs, va_loss_main, va_loss_x0 = [], [], []
    fid_train_hist, fid_val_hist = [], []
    best_val_loss, best_val_fid = float('inf'), float('inf')

    step = 0
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"[{mode}|{ds_key}] epoch {epoch+1}/{epochs}")
        for imgs, _ in pbar:
            imgs = imgs.to(device)
            t = torch.randint(0, ddpm.cfg.timesteps, (imgs.size(0),), device=device)
            if mode == "dsd":
                total, parts = deep_supervised_loss(model, ddpm, imgs, t,
                                                    w_aux_eps=dsd_w_aux_eps, w_aux_x0=dsd_w_aux_x0)
            else:
                total, parts = unified_loss(model, ddpm, imgs, t, multi_task=(mode=="multi"),
                                            w_x0=w_x0, w_consistency=w_consistency, multi_variant=multi_variant)
            optim.zero_grad(); total.backward(); optim.step()

            step += 1
            train_steps.append(step)
            tr_loss_main.append(float(parts["loss"]))
            if mode == "multi" or mode == "dsd":
                tr_loss_x0.append(float(parts["loss_x0"]))
                tr_loss_cons.append(float(parts["loss_cons"]))
                tr_loss_total.append(float(parts["loss_total"]))
                pbar.set_postfix({"loss": f"{parts['loss']:.4f}", "tot": f"{parts['loss_total']:.4f}"})
            else:
                pbar.set_postfix({"loss": f"{parts['loss']:.4f}"})

            if step % sample_every == 0:
                with torch.no_grad():
                    grid_path = images_dir / f"{file_prefix}_samples_step{step}.png"
                    _path, _ = sample(
                        model, ddpm,
                        shape=(n_sample, channels, img_size, img_size),
                        device=device, save_path=str(grid_path),
                    )

        val_metrics = evaluate_mse_unified(model, ddpm, val_loader, device, multi_task=(mode=="multi"))
        va_loss_main.append(float(val_metrics["loss"]))
        if mode == "multi":
            va_loss_x0.append(float(val_metrics["loss_x0"]))

        for d in [fid_train_real, fid_train_fake, fid_val_real, fid_val_fake]:
            for f in d.glob(f"{file_prefix}_*.png"): f.unlink()

        collected = 0; imgs_accum = []
        for imgs, _ in train_fid_loader:
            imgs_accum.append(imgs)
            collected += imgs.size(0)
            if collected >= fid_eval_images: break
        real_train = torch.cat(imgs_accum, dim=0)[:fid_eval_images]
        real_train = denorm(real_train, channels)
        dump_images(real_train, str(fid_train_real), prefix=f"{file_prefix}_real")

        collected = 0; imgs_accum = []
        for imgs, _ in val_loader:
            imgs_accum.append(imgs)
            collected += imgs.size(0)
            if collected >= fid_eval_images: break
        real_val = torch.cat(imgs_accum, dim=0)[:fid_eval_images]
        real_val = denorm(real_val, channels)
        dump_images(real_val, str(fid_val_real), prefix=f"{file_prefix}_real")

        _, fake_batch = sample(
            model, ddpm,
            shape=(min(fid_eval_images, n_sample), channels, img_size, img_size),
            device=device, save_path=str(images_dir / f"{file_prefix}_samples_trainfid_epoch{epoch+1}.png"),
        )
        fake_list = [denorm(fake_batch.cpu(), channels)]
        while sum(x.size(0) for x in fake_list) < fid_eval_images:
            _, fb = sample(
                model, ddpm,
                shape=(min(fid_eval_images - sum(x.size(0) for x in fake_list), n_sample),
                       channels, img_size, img_size),
                device=device, save_path=str(images_dir / "_tmp.png"),
            )
            fake_list.append(denorm(fb.cpu(), channels))
        fake_train = torch.cat(fake_list, dim=0)[:fid_eval_images]
        dump_images(fake_train, str(fid_train_fake), prefix=f"{file_prefix}_fake")

        _, fake_batch_val = sample(
            model, ddpm,
            shape=(min(fid_eval_images, n_sample), channels, img_size, img_size),
            device=device, save_path=str(images_dir / f"{file_prefix}_samples_valfid_epoch{epoch+1}.png"),
        )
        fake_list_val = [denorm(fake_batch_val.cpu(), channels)]
        while sum(x.size(0) for x in fake_list_val) < fid_eval_images:
            _, fb = sample(
                model, ddpm,
                shape=(min(fid_eval_images - sum(x.size(0) for x in fake_list_val), n_sample),
                       channels, img_size, img_size),
                device=device, save_path=str(images_dir / "_tmp2.png"),
            )
            fake_list_val.append(denorm(fb.cpu(), channels))
        fake_val = torch.cat(fake_list_val, dim=0)[:fid_eval_images]
        dump_images(fake_val, str(fid_val_fake), prefix=f"{file_prefix}_fake")

        fid_train = compute_fid(str(fid_train_real), str(fid_train_fake), device)
        fid_val   = compute_fid(str(fid_val_real),   str(fid_val_fake),   device)
        fid_train_hist.append(float(fid_train))
        fid_val_hist.append(float(fid_val))

        # Save best checkpoints by validation loss and FID
        current_val_loss = va_loss_main[-1]
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            torch.save({
                'state_dict': model.state_dict(),
                'mode': mode,
                'ds_key': ds_key,
                'base': base,
                'time_dim': time_dim,
                'channels': channels,
                'timesteps': timesteps,
                'beta_start': beta_start,
                'beta_end': beta_end,
            }, ckpt_dir / f"{file_prefix}_best_loss.pt")
        if float(fid_val) < best_val_fid:
            best_val_fid = float(fid_val)
            torch.save({
                'state_dict': model.state_dict(),
                'mode': mode,
                'ds_key': ds_key,
                'base': base,
                'time_dim': time_dim,
                'channels': channels,
                'timesteps': timesteps,
                'beta_start': beta_start,
                'beta_end': beta_end,
            }, ckpt_dir / f"{file_prefix}_best_fid.pt")

        if mode == "multi":
            print(f"[val] epoch {epoch+1}: loss={va_loss_main[-1]:.6f} | loss_x0={va_loss_x0[-1]:.6f} | "
                  f"FID_train={fid_train:.2f} | FID_val={fid_val:.2f}")
        else:
            print(f"[val] epoch {epoch+1}: loss={va_loss_main[-1]:.6f} | "
                  f"FID_train={fid_train:.2f} | FID_val={fid_val:.2f}")

        save_curves_unified_prefixed(
            metrics_dir,
            train_steps=train_steps,
            train_loss_main=tr_loss_main,
            val_epochs=list(range(1, len(va_loss_main)+1)),
            val_loss_main=va_loss_main,
            train_loss_x0=(tr_loss_x0 if mode=="multi" else None),
            train_loss_cons=(tr_loss_cons if mode=="multi" else None),
            train_loss_total=(tr_loss_total if mode=="multi" else None),
            val_loss_x0=(va_loss_x0 if mode=="multi" else None),
            fid_train=fid_train_hist,
            fid_val=fid_val_hist,
            file_prefix=file_prefix,
        )

    with torch.no_grad():
        path, _ = sample(
            model, ddpm,
            shape=(n_sample, channels, img_size, img_size),
            device=device, save_path=str(images_dir / f"{file_prefix}_samples_final.png"),
        )
    print(f"Saved final samples to {path}")

    save_curves_unified_prefixed(
        metrics_dir,
        train_steps=train_steps,
        train_loss_main=tr_loss_main,
        val_epochs=list(range(1, len(va_loss_main)+1)),
        val_loss_main=va_loss_main,
        train_loss_x0=(tr_loss_x0 if mode=="multi" else None),
        train_loss_cons=(tr_loss_cons if mode=="multi" else None),
        train_loss_total=(tr_loss_total if mode=="multi" else None),
        val_loss_x0=(va_loss_x0 if mode=="multi" else None),
        fid_train=fid_train_hist,
        fid_val=fid_val_hist,
        file_prefix=file_prefix,
    )


@torch.no_grad()
def evaluate_mse_unified(model, ddpm: DDPM, data_loader, device, multi_task: bool):
    model.eval()
    tot_eps, tot_x0, denom = 0.0, 0.0, 0
    for imgs, _ in data_loader:
        imgs = imgs.to(device)
        t = torch.randint(0, ddpm.cfg.timesteps, (imgs.size(0),), device=device)
        x_t, noise = ddpm.q_sample(imgs, t)
        preds = model(x_t, t.float())
        if multi_task:
            eps = preds["eps"]; x0p = preds["x0"]
            mse_eps = F.mse_loss(eps, noise, reduction='sum')
            mse_x0  = F.mse_loss(x0p, imgs, reduction='sum')
            tot_eps += mse_eps.item()
            tot_x0  += mse_x0.item()
        else:
            # Handle tensor or dict with 'eps'
            eps = preds["eps"] if isinstance(preds, dict) else preds
            mse_eps = F.mse_loss(eps, noise, reduction='sum')
            tot_eps += mse_eps.item()
        denom += imgs.numel()
    denom = max(1, denom)
    out = {"loss": tot_eps / denom}
    out["loss_x0"] = (tot_x0 / denom) if multi_task else None
    return out
