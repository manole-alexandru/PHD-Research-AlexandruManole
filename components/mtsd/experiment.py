from __future__ import annotations

# Entry point to run the experiments using split modules

from pathlib import Path
from datetime import datetime
import sys
import json
import argparse
import torch
import time
import warnings

# Suppress noisy deprecation from dependencies (TypedStorage -> UntypedStorage)
warnings.filterwarnings(
    "ignore",
    message=r".*TypedStorage is deprecated.*",
    category=UserWarning,
)

# Make sibling modules importable when running this file directly
CUR_DIR = Path(__file__).parent
if str(CUR_DIR) not in sys.path:
    sys.path.insert(0, str(CUR_DIR))

from training import train_unified, evaluate_mse_unified  # type: ignore
from data_utils import make_test_loader  # type: ignore
from diffusion import DDPM, DiffusionConfig  # type: ignore
from models import TinyUNet, DeepSupervisedUNet  # type: ignore
from metrics_utils import denorm, dump_images, compute_fid  # type: ignore
from sampling import sample  # type: ignore
from testing import run_post_training_testing  # type: ignore


if __name__ == "__main__":

    # Early GPU check
    try:
        _cuda = torch.cuda.is_available()
        _gpu_count = torch.cuda.device_count() if _cuda else 0
        _device_name = (
            torch.cuda.get_device_name(0) if (_cuda and _gpu_count > 0) else "CPU"
        )
        print(f"[Init] CUDA available: {_cuda} | GPUs: {_gpu_count} | Using: {_device_name}")
    except Exception as e:
        print(f"[Init] CUDA check failed: {e}")

    # CLI parameters
    parser = argparse.ArgumentParser(description="Run diffusion experiments on MNIST/CIFAR10")
    parser.add_argument(
        "--data",
        choices=["mnist", "cifar10", "cifar", "cifar100", "svhn", "celeba", "both", "all", "none"],
        default="both",
        help=(
            "Select dataset(s): mnist, cifar10, cifar100, svhn, celeba, both (mnist+cifar10), all (all supported), or none (eval-only)"
        ),
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        default=None,
        help="Optional experiment root directory to reuse across runs (e.g., to aggregate MNIST/CIFAR).",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        default=True,
        help="After training, perform post-train evaluation and plotting based only on saved files.",
    )
    parser.add_argument(
        "--keep-fid-images",
        action="store_true",
        default=True,
        help="Do not delete per-epoch/test FID image folders; keep PNGs for inspection.",
    )
    parser.add_argument(
        "--exp-number", "--exp_no",
        dest="exp_number",
        type=int,
        default=1,
        help="Experiment number/id to use when creating EXP_ROOT (ignored if --exp-dir is provided).",
    )
    args = parser.parse_args()

    data_choice = args.data.lower()
    if data_choice in ("cifar10", "cifar"):
        DATASETS = ["cifar10"]
    elif data_choice in ("mnist", "cifar100", "svhn", "celeba"):
        DATASETS = [data_choice]
    elif data_choice == "both":
        DATASETS = ["cifar10", "mnist"]
    elif data_choice == "all":
        DATASETS = ["cifar10", "mnist", "cifar100", "svhn", "celeba"]
    elif data_choice == "none":
        DATASETS = []
    else:
        DATASETS = ["cifar10", "mnist"]

    # Centralized constants (change once here)
    EPOCHS = 3
    WITH_DSD = False
    TIMESTEPS = 200
    N_SAMPLE = 64
    BATCH_SIZE = 128
    SAMPLE_EVERY = 500
    EXP_NO = int(args.exp_number)  # experiment number/id

    # Experiment folder: allow reusing a fixed directory via --exp-dir for cross-run aggregation
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    EXP_BASE = Path("/content/runs/") # Path("C:/Mano/profession/phd/PHD-Research-AlexandruManole/components/mtsd/runs/") # Path("/content/drive/MyDrive/prototypes/mtsd_exp")
    if args.exp_dir:
        EXP_ROOT = Path(args.exp_dir)
    else:
        EXP_ROOT = EXP_BASE / f"experiment-{EXP_NO}" # -{timestamp}"
    (EXP_ROOT / "images").mkdir(parents=True, exist_ok=True)
    (EXP_ROOT / "metrics").mkdir(parents=True, exist_ok=True)
    (EXP_ROOT / "fid").mkdir(parents=True, exist_ok=True)

    # Save config to experiment folder
    cfg = {
        "EXP_NO": EXP_NO,
        "EPOCHS": EPOCHS,
        "TIMESTEPS": TIMESTEPS,
        "N_SAMPLE": N_SAMPLE,
        "SAMPLE_EVERY": SAMPLE_EVERY,
        "BATCH_SIZE": BATCH_SIZE,
        "LR": 2e-4,
        "BETA_START": 1e-4,
        "BETA_END": 0.02,
        "BASE": 32,
        "TIME_DIM": 128,
        "VAL_SPLIT": 0.05,
        "FID_EVAL_IMAGES": 2048,
        "W_X0": 1.0,
        "W_CONSISTENCY": 0.1,
        "WITH_DSD": WITH_DSD,
    }
    try:
        cfg_path = EXP_ROOT / "config.json"
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        assert cfg_path.exists() and cfg_path.stat().st_size > 0, f"Config not saved: {cfg_path}"
    except Exception as e:
        print(f"Warning: failed to write config.json: {e}")

    # Runs: Single-task, Multi-task, and optional Deep Supervised Diffusion for each dataset
    for ds in DATASETS:
        # Single-task
        train_unified(
            save_root=str(EXP_ROOT),
            experiment_dir=EXP_ROOT,
            mode="single",
            data=ds,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            timesteps=TIMESTEPS,
            n_sample=N_SAMPLE,
            sample_every=SAMPLE_EVERY,
            keep_fid_images=bool(args.keep_fid_images),
        )
        # Multi-task (named variant: eps_x0_consistency)
        train_unified(
            save_root=str(EXP_ROOT),
            experiment_dir=EXP_ROOT,
            mode="multi",
            data=ds,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            timesteps=TIMESTEPS,
            n_sample=N_SAMPLE,
            sample_every=SAMPLE_EVERY,
            w_x0=cfg["W_X0"],
            w_consistency=cfg["W_CONSISTENCY"],
            multi_variant="eps_x0_consistency",
            keep_fid_images=bool(args.keep_fid_images),
        )
        # Deep Supervised Diffusion (optional)
        if WITH_DSD:
            train_unified(
                save_root=str(EXP_ROOT),
                experiment_dir=EXP_ROOT,
                mode="dsd",
                data=ds,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                timesteps=TIMESTEPS,
                n_sample=N_SAMPLE,
                sample_every=SAMPLE_EVERY,
                keep_fid_images=bool(args.keep_fid_images),
            )

    # After all runs, create combined comparison plots (val loss and FID) per dataset
    # Note: this section operates solely on saved files (metrics CSVs, checkpoints),
    #       not on in-memory variables, to support aggregating results across multiple runs.
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics_dir = EXP_ROOT / "metrics"

    def _read_val_metrics(prefix: str):
        path = metrics_dir / f"{prefix}_val_metrics.csv"
        epochs, losses, loss_x0, fid_train, fid_val = [], [], [], [], []
        if not path.exists():
            return epochs, losses, loss_x0, fid_train, fid_val
        with open(path, "r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    epochs.append(int(float(row.get("epoch", 0))))
                    losses.append(float(row.get("loss", "nan")))
                    if "loss_x0" in row and row.get("loss_x0", "") != "":
                        loss_x0.append(float(row.get("loss_x0", "nan")))
                    if "fid_train" in row and row.get("fid_train", "") != "":
                        fid_train.append(float(row.get("fid_train", "nan")))
                    if "fid_val" in row and row.get("fid_val", "") != "":
                        fid_val.append(float(row.get("fid_val", "nan")))
                except Exception:
                    continue
        return epochs, losses, loss_x0, fid_train, fid_val

    # If --eval is not set, skip all post-train evaluation steps
    # if not args.eval:
    #    sys.exit(0)

    # Per-run: FID value per epoch (train & validation if available)
    for ds in DATASETS:
        for mode, color in [("single", "tab:blue"), ("multi", "tab:orange")]:
            prefix = f"{ds}_{mode}"
            ep, _, _, fid_tr, fid_v = _read_val_metrics(prefix)
            if not ep or (not fid_tr and not fid_v):
                continue
            plt.figure()
            if fid_tr:
                plt.plot(ep[:len(fid_tr)], fid_tr, label="FID (train)", color=color, linestyle="-")
            if fid_v:
                plt.plot(ep[:len(fid_v)], fid_v, label="FID (val)", color=color, linestyle=":")
            plt.xlabel("epoch"); plt.ylabel("FID"); plt.title(f"FID per Epoch - {ds} ({mode})"); plt.legend(); plt.tight_layout()
            out_fid = metrics_dir / f"{prefix}_fid_per_epoch.png"
            plt.savefig(out_fid, dpi=150); plt.close()
            time.sleep(0.1)
            assert out_fid.exists() and out_fid.stat().st_size > 0, f"Plot not saved: {out_fid}"

    # Combined single vs multi: one figure with (left) FID evolution, (right) loss evolution
    for ds in DATASETS:
        single = _read_val_metrics(f"{ds}_single")
        multi  = _read_val_metrics(f"{ds}_multi")
        ep_s, loss_s, _, fid_tr_s, fid_v_s = single
        ep_m, loss_m, loss_m_x0, fid_tr_m, fid_v_m = multi
        if not ep_s or not ep_m:
            continue
        # Choose FID series preference: val if available else train
        fid_s = fid_v_s if fid_v_s else fid_tr_s
        fid_m = fid_v_m if fid_v_m else fid_tr_m
        if not fid_s and not fid_m:
            # still create the figure to show loss comparison
            pass
        # Build the combined figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        ax_fid, ax_loss = axes
        # FID evolution
        plotted_any = False
        if fid_s:
            ax_fid.plot(ep_s[:len(fid_s)], fid_s, label="single", color="tab:blue")
            plotted_any = True
        if fid_m:
            ax_fid.plot(ep_m[:len(fid_m)], fid_m, label="multi", color="tab:orange")
            plotted_any = True
        ax_fid.set_xlabel("epoch"); ax_fid.set_ylabel("FID"); ax_fid.set_title(f"FID per Epoch - {ds}")
        if plotted_any:
            ax_fid.legend()
        # Loss evolution (validation loss); ensure identical objective (eps) is included
        plotted_any = False
        if loss_s:
            ax_loss.plot(ep_s[:len(loss_s)], loss_s, label="single (val eps)", color="tab:blue")
            plotted_any = True
        if loss_m:
            ax_loss.plot(ep_m[:len(loss_m)], loss_m, label="multi (val eps)", color="tab:orange")
            plotted_any = True
        # Optionally also show multi's x0 loss if present
        if loss_m_x0:
            ax_loss.plot(ep_m[:len(loss_m_x0)], loss_m_x0, label="multi (val x0)", color="tab:orange", linestyle=":")
        ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("loss"); ax_loss.set_title(f"Loss per Epoch - {ds}")
        if plotted_any:
            ax_loss.legend()
        fig.tight_layout()
        out_cmp = metrics_dir / f"{ds}_single_vs_multi_fid_and_loss.png"
        fig.savefig(out_cmp, dpi=150)
        plt.close(fig)
        time.sleep(0.1)
        assert out_cmp.exists() and out_cmp.stat().st_size > 0, f"Plot not saved: {out_cmp}"

    # Evaluate best checkpoints on test set (extracted)
    run_post_training_testing(
        EXP_ROOT,
        DATASETS,
        with_dsd=WITH_DSD,
        fid_eval_images=int(cfg.get("FID_EVAL_IMAGES", 1024)),
        keep_fid_images=bool(args.keep_fid_images),
    )
