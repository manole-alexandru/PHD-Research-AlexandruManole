from __future__ import annotations

# Entry point to run the experiments using split modules

from pathlib import Path
from datetime import datetime
import sys
import json
import argparse
import torch

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
    # CLI parameters
    parser = argparse.ArgumentParser(description="Run diffusion experiments on MNIST/CIFAR10")
    parser.add_argument(
        "--data",
        choices=["mnist", "cifar10", "cifar", "both", "none"],
        default="both",
        help="Select dataset(s) to train: mnist, cifar10, both, or none (eval-only)",
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
        help="After training, perform post-train evaluation and plotting based only on saved files.",
    )
    args = parser.parse_args()

    data_choice = args.data.lower()
    if data_choice in ("cifar10", "cifar"):
        DATASETS = ["cifar10"]
    elif data_choice == "mnist":
        DATASETS = ["mnist"]
    elif data_choice == "none":
        DATASETS = []
    else:
        DATASETS = ["cifar10", "mnist"]

    # Centralized constants (change once here)
    EPOCHS = 3
    WITH_DSD = False
    TIMESTEPS = 200
    N_SAMPLE = 64
    SAMPLE_EVERY = 500
    EXP_NO = 1  # experiment number/id

    # Experiment folder: allow reusing a fixed directory via --exp-dir for cross-run aggregation
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    EXP_BASE = Path("/content/drive/MyDrive/prototypes/mtsd_exp")
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
        "BATCH_SIZE": 128,
        "LR": 2e-4,
        "BETA_START": 1e-4,
        "BETA_END": 0.02,
        "BASE": 32,
        "TIME_DIM": 128,
        "VAL_SPLIT": 0.05,
        "FID_EVAL_IMAGES": 256,
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
            timesteps=TIMESTEPS,
            n_sample=N_SAMPLE,
            sample_every=SAMPLE_EVERY,
        )
        # Multi-task (named variant: eps_x0_consistency)
        train_unified(
            save_root=str(EXP_ROOT),
            experiment_dir=EXP_ROOT,
            mode="multi",
            data=ds,
            epochs=EPOCHS,
            timesteps=TIMESTEPS,
            n_sample=N_SAMPLE,
            sample_every=SAMPLE_EVERY,
            w_x0=cfg["W_X0"],
            w_consistency=cfg["W_CONSISTENCY"],
            multi_variant="eps_x0_consistency",
        )
        # Deep Supervised Diffusion (optional)
        if WITH_DSD:
            train_unified(
                save_root=str(EXP_ROOT),
                experiment_dir=EXP_ROOT,
                mode="dsd",
                data=ds,
                epochs=EPOCHS,
                timesteps=TIMESTEPS,
                n_sample=N_SAMPLE,
                sample_every=SAMPLE_EVERY,
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
        epochs, losses, fid_val = [], [], []
        if not path.exists():
            return epochs, losses, fid_val
        with open(path, "r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    epochs.append(int(float(row.get("epoch", 0))))
                    losses.append(float(row.get("loss", "nan")))
                    if "fid_val" in row:
                        fid_val.append(float(row.get("fid_val", "nan")))
                except Exception:
                    continue
        return epochs, losses, fid_val

    # If --eval is not set, skip all post-train evaluation steps
    if not args.eval:
        sys.exit(0)

    # Combined comparisons (single vs multi): FID-only and Loss-only
    # Always consider both datasets here; functions handle missing files gracefully.
    for ds in ["mnist", "cifar10"]:
        prefixes = {
            "single": f"{ds}_single",
            "multi": f"{ds}_multi",
        }
        series = {k: _read_val_metrics(v) for k, v in prefixes.items()}

        # FID-only (single vs multi)
        plt.figure()
        for label, color in [("single", "tab:blue"), ("multi", "tab:orange")]:
            ep, _, fidv = series.get(label, ([], [], []))
            if ep and fidv:
                plt.plot(ep, fidv, label=label, color=color)
        plt.xlabel("epoch"); plt.ylabel("FID (val)"); plt.title(f"FID (val) - {ds} (single vs multi)"); plt.legend(); plt.tight_layout()
        out_fid_only = metrics_dir / f"{ds}_single_vs_multi_val_fid.png"
        plt.savefig(out_fid_only, dpi=150); plt.close()
        try:
            assert out_fid_only.exists() and out_fid_only.stat().st_size > 0, f"Plot not saved: {out_fid_only}"
        except Exception as e:
            raise AssertionError(f"Failed to save FID-only plot '{out_fid_only}': {e}")

        # Loss-only (single vs multi)
        plt.figure()
        for label, color in [("single", "tab:blue"), ("multi", "tab:orange")]:
            ep, loss, _ = series.get(label, ([], [], []))
            if ep and loss:
                plt.plot(ep, loss, label=label, color=color)
        plt.xlabel("epoch"); plt.ylabel("val loss"); plt.title(f"Validation Loss - {ds} (single vs multi)"); plt.legend(); plt.tight_layout()
        out_loss_only = metrics_dir / f"{ds}_single_vs_multi_val_loss.png"
        plt.savefig(out_loss_only, dpi=150); plt.close()
        try:
            assert out_loss_only.exists() and out_loss_only.stat().st_size > 0, f"Plot not saved: {out_loss_only}"
        except Exception as e:
            raise AssertionError(f"Failed to save loss-only plot '{out_loss_only}': {e}")

    # Separate comparisons (val loss and FID), optionally including DSD
    for ds in ["mnist", "cifar10"]:
        prefixes = {
            "single": f"{ds}_single",
            "multi": f"{ds}_multi",
        }
        if WITH_DSD:
            prefixes["dsd"] = f"{ds}_dsd"
        series = {k: _read_val_metrics(v) for k, v in prefixes.items()}

        # Val loss comparison
        plt.figure()
        labels_colors = [("single", "tab:blue"), ("multi", "tab:orange")]
        if WITH_DSD:
            labels_colors.append(("dsd", "tab:green"))
        for label, color in labels_colors:
            ep, loss, _ = series.get(label, ([], [], []))
            if ep and loss:
                plt.plot(ep, loss, label=label, color=color)
        plt.xlabel("epoch"); plt.ylabel("val loss"); plt.title(f"Validation Loss - {ds}"); plt.legend(); plt.tight_layout()
        out_val_loss = metrics_dir / f"{ds}_compare_val_loss.png"
        plt.savefig(out_val_loss, dpi=150); plt.close()
        try:
            assert out_val_loss.exists() and out_val_loss.stat().st_size > 0, f"Plot not saved: {out_val_loss}"
        except Exception as e:
            raise AssertionError(f"Failed to save val-loss comparison plot '{out_val_loss}': {e}")

        # FID (val) comparison
        plt.figure()
        for label, color in labels_colors:
            ep, _, fidv = series.get(label, ([], [], []))
            if ep and fidv:
                plt.plot(ep, fidv, label=label, color=color)
        plt.xlabel("epoch"); plt.ylabel("FID (val)"); plt.title(f"FID (val) - {ds}"); plt.legend(); plt.tight_layout()
        out_fid_val = metrics_dir / f"{ds}_compare_fid_val.png"
        plt.savefig(out_fid_val, dpi=150); plt.close()
        try:
            assert out_fid_val.exists() and out_fid_val.stat().st_size > 0, f"Plot not saved: {out_fid_val}"
        except Exception as e:
            raise AssertionError(f"Failed to save FID-val comparison plot '{out_fid_val}': {e}")

    # Evaluate best checkpoints on test set (extracted)
    run_post_training_testing(EXP_ROOT, ["mnist", "cifar10"], with_dsd=WITH_DSD, fid_eval_images=1024)


# TODO: After single-task, multi-task and dds were trained, plot the train losses and val losses in the same plot + fid evolutions
