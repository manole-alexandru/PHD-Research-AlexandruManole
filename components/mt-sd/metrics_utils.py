from __future__ import annotations
from pathlib import Path
import csv
import torch
from torchvision import utils as vutils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def dump_images(tensor_bchw, out_dir: str, prefix: str = "img"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(tensor_bchw):
        fp = out / f"{prefix}_{i:05d}.png"
        vutils.save_image(img, fp)
        try:
            assert fp.exists() and fp.stat().st_size > 0, f"Image not saved: {fp}"
        except Exception as e:
            raise AssertionError(f"Failed to save image '{fp}': {e}")


def compute_fid(real_dir: str, fake_dir: str, device: torch.device):
    try:
        from torch_fidelity import calculate_metrics
    except Exception:
        print("torch-fidelity not found. Install with: pip install torch-fidelity")
        return float('nan')
    metrics = calculate_metrics(
        input1=real_dir, input2=fake_dir,
        fid=True, isc=False, kid=False,
        cuda=device.type == 'cuda', batch_size=64, verbose=False,
    )
    return float(metrics.get('frechet_inception_distance', float('nan')))


def ema_series(values, decay=0.98):
    if not values: return []
    out, m = [], values[0]
    for v in values:
        m = decay * m + (1 - decay) * v
        out.append(m)
    return out


def save_curves_unified_prefixed(
    out_dir: Path,
    train_steps,
    train_loss_main, val_epochs, val_loss_main,
    train_loss_x0=None, train_loss_cons=None, train_loss_total=None,
    val_loss_x0=None,
    fid_train=None, fid_val=None,
    file_prefix: str | None = None,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = (file_prefix + "_") if file_prefix else ""

    # CSVs
    train_csv = out_dir / f"{prefix}train_losses.csv"
    with open(train_csv, "w", newline="") as f:
        w = csv.writer(f)
        header = ["step", "loss"]
        if train_loss_x0 is not None: header += ["loss_x0"]
        if train_loss_cons is not None: header += ["loss_cons"]
        if train_loss_total is not None: header += ["loss_total"]
        w.writerow(header)
        for i, s in enumerate(train_steps):
            row = [s, train_loss_main[i]]
            if train_loss_x0 is not None: row += [train_loss_x0[i]]
            if train_loss_cons is not None: row += [train_loss_cons[i]]
            if train_loss_total is not None: row += [train_loss_total[i]]
            w.writerow(row)

    val_csv = out_dir / f"{prefix}val_metrics.csv"
    with open(val_csv, "w", newline="") as f:
        w = csv.writer(f)
        header = ["epoch", "loss"]
        if val_loss_x0 is not None: header += ["loss_x0"]
        if fid_train is not None: header += ["fid_train"]
        if fid_val is not None: header += ["fid_val"]
        w.writerow(header)
        for j, ep in enumerate(val_epochs):
            row = [ep, val_loss_main[j]]
            if val_loss_x0 is not None: row += [val_loss_x0[j]]
            if fid_train is not None: row += [fid_train[j]]
            if fid_val is not None: row += [fid_val[j]]
            w.writerow(row)

    # Plots: train loss main
    # Ensure CSVs were written
    try:
        assert train_csv.exists() and train_csv.stat().st_size > 0, f"Train CSV not saved: {train_csv}"
        assert val_csv.exists() and val_csv.stat().st_size > 0, f"Val CSV not saved: {val_csv}"
    except Exception as e:
        raise AssertionError(f"Failed writing CSV metrics: {e}")

    plt.figure()
    plt.plot(train_steps, train_loss_main, label="loss (raw)")
    plt.plot(train_steps, ema_series(train_loss_main), label="loss (EMA)")
    plt.xlabel("step"); plt.ylabel("loss"); plt.title("Training loss (main)"); plt.legend(); plt.tight_layout()
    pl_main = out_dir / f"{prefix}training_loss_main.png"
    plt.savefig(pl_main, dpi=150); plt.close()
    try:
        assert pl_main.exists() and pl_main.stat().st_size > 0, f"Plot not saved: {pl_main}"
    except Exception as e:
        raise AssertionError(f"Failed saving training loss main plot: {e}")

    # Optional components
    if train_loss_x0 is not None or train_loss_cons is not None or train_loss_total is not None:
        plt.figure()
        if train_loss_x0 is not None:   plt.plot(train_steps[:len(train_loss_x0)], ema_series(train_loss_x0), label="loss_x0 (EMA)")
        if train_loss_cons is not None: plt.plot(train_steps[:len(train_loss_cons)], ema_series(train_loss_cons), label="loss_cons (EMA)")
        if train_loss_total is not None:plt.plot(train_steps[:len(train_loss_total)], ema_series(train_loss_total), label="loss_total (EMA)")
        plt.xlabel("step"); plt.ylabel("loss"); plt.title("Training loss components"); plt.legend(); plt.tight_layout()
        pl_comp = out_dir / f"{prefix}training_loss_components.png"
        plt.savefig(pl_comp, dpi=150); plt.close()
        try:
            assert pl_comp.exists() and pl_comp.stat().st_size > 0, f"Plot not saved: {pl_comp}"
        except Exception as e:
            raise AssertionError(f"Failed saving training loss components plot: {e}")

    # Validation losses
    plt.figure()
    plt.plot(val_epochs, val_loss_main, label="val loss (eps)")
    if val_loss_x0 is not None: plt.plot(val_epochs, val_loss_x0, label="val loss_x0")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Validation losses"); plt.legend(); plt.tight_layout()
    pl_val = out_dir / f"{prefix}val_losses.png"
    plt.savefig(pl_val, dpi=150); plt.close()
    try:
        assert pl_val.exists() and pl_val.stat().st_size > 0, f"Plot not saved: {pl_val}"
    except Exception as e:
        raise AssertionError(f"Failed saving validation losses plot: {e}")

    # FID curves
    if fid_train is not None or fid_val is not None:
        plt.figure()
        if fid_train is not None: plt.plot(val_epochs, fid_train, label="FID (train)")
        if fid_val is not None:   plt.plot(val_epochs, fid_val,   label="FID (val)")
        plt.xlabel("epoch"); plt.ylabel("FID"); plt.title("FID over epochs"); plt.legend(); plt.tight_layout()
        pl_fid = out_dir / f"{prefix}fid.png"
        plt.savefig(pl_fid, dpi=150); plt.close()
        try:
            assert pl_fid.exists() and pl_fid.stat().st_size > 0, f"Plot not saved: {pl_fid}"
        except Exception as e:
            raise AssertionError(f"Failed saving FID plot: {e}")


def denorm(x, channels):
    x = (x.clamp(-1, 1) + 1) * 0.5
    if channels == 1:
        x = x.repeat(1, 3, 1, 1)
    return x
