from __future__ import annotations

# Entry point to run the experiments using split modules

from pathlib import Path
from datetime import datetime
import sys
import json

# Make sibling modules importable when running this file directly
CUR_DIR = Path(__file__).parent
if str(CUR_DIR) not in sys.path:
    sys.path.insert(0, str(CUR_DIR))

from training import train_unified  # type: ignore


if __name__ == "__main__":
    # Centralized constants (change once here)
    EPOCHS = 10
    TIMESTEPS = 200
    N_SAMPLE = 64
    SAMPLE_EVERY = 500
    EXP_NO = 1  # experiment number/id

    # Single experiment folder for all runs
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    EXP_ROOT = Path("runs") / f"experiment-{EXP_NO}-{timestamp}"
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
        "FID_EVAL_IMAGES": 1024,
        "W_X0": 1.0,
        "W_CONSISTENCY": 0.1,
    }
    try:
        with open(EXP_ROOT / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        print(f"Warning: failed to write config.json: {e}")

    # Single-task runs: MNIST and CIFAR10
    for ds in ["mnist", "cifar10"]:
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

    # Multi-task runs: MNIST and CIFAR10
    for ds in ["mnist", "cifar10"]:
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
        )

