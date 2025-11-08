from __future__ import annotations

from pathlib import Path
from typing import List
import torch
import csv as _csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import shutil

from data_utils import make_test_loader  # type: ignore
from diffusion import DDPM, DiffusionConfig  # type: ignore
from models import TinyUNet, DeepSupervisedUNet  # type: ignore
from metrics_utils import denorm, dump_images, compute_fid  # type: ignore
from sampling import sample  # type: ignore


def run_post_training_testing(
    exp_root: Path,
    datasets: List[str],
    with_dsd: bool = False,
    fid_eval_images: int = 1024,
    keep_fid_images: bool = False,
) -> Path:
    """
    Evaluate best checkpoints on the test set (MSE + FID), save CSV summary and FID bar plots.

    Args:
        exp_root: Root experiment directory (under EXP_BASE).
        datasets: List of dataset keys to evaluate (e.g., ["mnist", "cifar10"]).
        with_dsd: Whether to include Deep Supervised Diffusion checkpoints if available.
        fid_eval_images: Number of images to use for FID evaluation.

    Returns:
        Path to the saved CSV summary.
    """
    metrics_dir = exp_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    fid_dir = exp_root / "fid"
    fid_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = exp_root / "checkpoints"

    results = []
    for ds in datasets:
        ds_key = ds
        channels = 1 if ds_key == "mnist" else 3
        img_size = 28 if ds_key == "mnist" else 32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_loader = make_test_loader(ds_key, batch_size=128, img_size=img_size, channels=channels)
        modes = ["single", "multi"] + (["dsd"] if with_dsd else [])

        for mode in modes:
            file_prefix = f"{ds_key}_{mode}"
            for tag in ["best_loss", "best_fid"]:
                ckpt_path = ckpt_dir / f"{file_prefix}_{tag}.pt"
                if not ckpt_path.exists():
                    continue

                payload = torch.load(ckpt_path, map_location="cpu")
                # Build model per mode
                if mode == "dsd":
                    model = DeepSupervisedUNet(
                        in_channels=channels,
                        base=payload.get('base', 32),
                        time_dim=payload.get('time_dim', 128),
                    ).to(device)
                else:
                    model = TinyUNet(
                        in_channels=channels,
                        base=payload.get('base', 32),
                        time_dim=payload.get('time_dim', 128),
                        multi_task=(mode == "multi"),
                    ).to(device)
                model.load_state_dict(payload['state_dict'])
                model.eval()
                ddpm = DDPM(
                    DiffusionConfig(
                        timesteps=payload.get('timesteps', 200),
                        beta_start=payload.get('beta_start', 1e-4),
                        beta_end=payload.get('beta_end', 0.02),
                    )
                ).to(device)

                # MSE on test set
                from training import evaluate_mse_unified  # local import to avoid cycles  # type: ignore
                metrics = evaluate_mse_unified(model, ddpm, test_loader, device, multi_task=(mode == "multi"))
                mse_loss = float(metrics["loss"]) if metrics and ("loss" in metrics) else float('nan')

                # FID on test set: prepare real test and fake samples
                # Use a temporary per-evaluation epoch-like directory and clean it after FID
                ep_dir = fid_dir / f"{file_prefix}_test" / "epoch_eval"
                test_real_dir = ep_dir / "real"
                test_fake_dir = ep_dir / "fake"
                for d in [test_real_dir, test_fake_dir]:
                    d.mkdir(parents=True, exist_ok=True)
                    # cleanup previous (only if not preserving)
                    if not keep_fid_images:
                        for f in d.glob("*.png"):
                            f.unlink()

                collected = 0
                imgs_accum = []
                for imgs, _ in test_loader:
                    imgs_accum.append(imgs)
                    collected += imgs.size(0)
                    if collected >= fid_eval_images:
                        break
                real_test = torch.cat(imgs_accum, dim=0)[:fid_eval_images]
                real_test = denorm(real_test, channels)
                dump_images(real_test, str(test_real_dir), prefix="real")

                # fake
                n_sample = 64
                fake_list = []
                while sum(x.size(0) for x in fake_list) < fid_eval_images:
                    n = min(fid_eval_images - sum(x.size(0) for x in fake_list), n_sample)
                    _, fb = sample(
                        model,
                        ddpm,
                        shape=(n, channels, img_size, img_size),
                        device=device,
                        save_path=str((exp_root / "images") / "_tmp_test.png"),
                    )
                    fake_list.append(denorm(fb.cpu(), channels))
                fake_test = torch.cat(fake_list, dim=0)[:fid_eval_images]
                dump_images(fake_test, str(test_fake_dir), prefix="fake")

                fid_test = compute_fid(str(test_real_dir), str(test_fake_dir), device)
                results.append({
                    "dataset": ds_key,
                    "mode": mode,
                    "checkpoint": tag,
                    "mse_loss": mse_loss,
                    "fid_test": float(fid_test),
                })
                # Clean up temporary directories (optional)
                if not keep_fid_images:
                    try:
                        shutil.rmtree(ep_dir, ignore_errors=True)
                    except Exception:
                        pass
                else:
                    print(f"[fid|test] kept evaluation images at {ep_dir}")

    # Save results CSV
    out_csv = metrics_dir / "test_summary.csv"
    with open(out_csv, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=["dataset", "mode", "checkpoint", "mse_loss", "fid_test"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    time.sleep(0.1)
    assert out_csv.exists() and out_csv.stat().st_size > 0, f"Test summary CSV not saved: {out_csv}"

    # Bar plots of FID per dataset
    for ds in datasets:
        items = [r for r in results if r["dataset"] == ds]
        if not items:
            continue
        labels = [f"{r['mode']}_{r['checkpoint']}" for r in items]
        values = [r['fid_test'] for r in items]
        plt.figure(figsize=(10, 4))
        plt.bar(
            labels,
            values,
            color=["tab:blue" if 'single' in lb else ("tab:orange" if 'multi' in lb else "tab:green") for lb in labels],
        )
        plt.ylabel("FID (test)"); plt.title(f"Test FID - {ds}"); plt.xticks(rotation=30, ha='right'); plt.tight_layout()
        out_bar = metrics_dir / f"{ds}_test_fid_bar.png"
        plt.savefig(out_bar, dpi=150); plt.close()
        time.sleep(0.1)
        assert out_bar.exists() and out_bar.stat().st_size > 0, f"Bar plot not saved: {out_bar}"

    return out_csv
