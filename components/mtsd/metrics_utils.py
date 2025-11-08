from __future__ import annotations
from pathlib import Path
import math
import warnings
import csv
import torch
from torchvision import utils as vutils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from metrics.metrics_logger import MetricsLogger


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
        from cleanfid import fid as _cleanfid
    except Exception:
        print("clean-fid not found. Install with: pip install clean-fid")
        return float('nan')
    # Basic sanity: ensure there are images present in both folders
    try:
        r_count = len(list(Path(real_dir).glob('*.png')))
        f_count = len(list(Path(fake_dir).glob('*.png')))
        if r_count == 0 or f_count == 0:
            print(f"[fid] Skipping FID: empty dirs real={r_count} fake={f_count} at {real_dir} | {fake_dir}")
            return float('nan')
    except Exception:
        pass

    def _calc(dev_str: str, batch_size: int = 64):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*TypedStorage is deprecated.*",
                category=UserWarning,
            )
            return float(_cleanfid.compute_fid(
                real_dir,
                fake_dir,
                device=dev_str,
                num_workers=0,
                batch_size=batch_size,
            ))

    use_cuda = (device.type == 'cuda') and torch.cuda.is_available()

    # First try on GPU (if available)
    if use_cuda:
        try:
            fid_gpu = _calc('cuda', batch_size=64)
        except Exception as e:
            print(f"[fid] GPU run failed ({e}); falling back to CPU.")
            fid_gpu = float('nan')

        if not (math.isfinite(fid_gpu)):
            # Workaround: some PyTorch/torch-fidelity combos fail to collate GPU batches
            # due to non-resizable storages. Retry on GPU with batch_size=1.
            try:
                fid_gpu_b1 = _calc('cuda', batch_size=1)
                if math.isfinite(fid_gpu_b1):
                    print("[fid] GPU retry with batch_size=1 succeeded; using GPU FID.")
                    return fid_gpu_b1
            except Exception as e:
                print(f"[fid] GPU retry (batch_size=1) failed: {e}")

            # Retry on CPU
            try:
                fid_cpu = _calc('cpu')
                if not math.isfinite(fid_cpu):
                    print("[fid] CPU fallback also returned NaN/Inf FID.")
                else:
                    print("[fid] CPU fallback succeeded; using CPU FID.")
                return fid_cpu
            except Exception as e:
                print(f"[fid] CPU fallback failed: {e}")
                return float('nan')
        else:
            return fid_gpu

    # CPU path
    try:
        fid = _calc('cpu')
        return fid
    except Exception as e:
        print(f"[fid] CPU FID calculation failed: {e}")
        return float('nan')


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
    train_loss_eps=None,
    train_loss_x0=None, train_loss_cons=None, train_loss_total=None,
    val_loss_x0=None,
    fid_train=None, fid_val=None,
    file_prefix: str | None = None,
):
    logger = MetricsLogger(Path(out_dir), file_prefix)
    logger.save_all(
        train_steps=train_steps,
        train_loss_main=train_loss_main,
        val_epochs=val_epochs,
        val_loss_main=val_loss_main,
        train_loss_eps=train_loss_eps,
        train_loss_x0=train_loss_x0,
        train_loss_cons=train_loss_cons,
        train_loss_total=train_loss_total,
        val_loss_x0=val_loss_x0,
        fid_train=fid_train,
        fid_val=fid_val,
    )


def denorm(x, channels):
    x = (x.clamp(-1, 1) + 1) * 0.5
    if channels == 1:
        x = x.repeat(1, 3, 1, 1)
    return x
