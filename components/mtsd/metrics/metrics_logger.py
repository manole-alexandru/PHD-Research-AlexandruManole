from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
try:
    import torch
except Exception:
    torch = None  # type: ignore


def ema_series(values: Sequence[float], decay: float = 0.98):
    if not values:
        return []
    out, m = [], values[0]
    for v in values:
        m = decay * m + (1 - decay) * v
        out.append(m)
    return out


class MetricsLogger:
    def __init__(self, out_dir: Path, file_prefix: Optional[str] = None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = (file_prefix + "_") if file_prefix else ""

    # --------------- CSV ---------------
    def write_train_csv(
        self,
        train_steps: Sequence[int],
        train_loss_main: Sequence[float],
        train_loss_eps: Optional[Sequence[float]] = None,
        train_loss_x0: Optional[Sequence[float]] = None,
        train_loss_cons: Optional[Sequence[float]] = None,
        train_loss_total: Optional[Sequence[float]] = None,
    ) -> Path:
        train_csv = self.out_dir / f"{self.prefix}train_losses.csv"
        with open(train_csv, "w", newline="") as f:
            w = csv.writer(f)
            header = ["epoch", "loss"]
            if train_loss_eps is not None: header += ["loss_eps"]
            if train_loss_x0 is not None: header += ["loss_x0"]
            if train_loss_cons is not None: header += ["loss_cons"]
            if train_loss_total is not None: header += ["loss_total"]
            w.writerow(header)
            for i, s in enumerate(train_steps):
                row = [s, train_loss_main[i]]
                if train_loss_eps is not None: row += [train_loss_eps[i]]
                if train_loss_x0 is not None: row += [train_loss_x0[i]]
                if train_loss_cons is not None: row += [train_loss_cons[i]]
                if train_loss_total is not None: row += [train_loss_total[i]]
                w.writerow(row)
        time.sleep(0.1)
        assert train_csv.exists() and train_csv.stat().st_size > 0, f"Train CSV not saved: {train_csv}"
        return train_csv

    def write_val_csv(
        self,
        val_epochs: Sequence[int],
        val_loss_main: Sequence[float],
        val_loss_x0: Optional[Sequence[float]] = None,
        fid_train: Optional[Sequence[float]] = None,
        fid_val: Optional[Sequence[float]] = None,
        # Optional multi-task variant FIDs
        fid_train_eps: Optional[Sequence[float]] = None,
        fid_train_x0: Optional[Sequence[float]] = None,
        fid_train_combined: Optional[Sequence[float]] = None,
        fid_val_eps: Optional[Sequence[float]] = None,
        fid_val_x0: Optional[Sequence[float]] = None,
        fid_val_combined: Optional[Sequence[float]] = None,
    ) -> Path:
        val_csv = self.out_dir / f"{self.prefix}val_metrics.csv"
        with open(val_csv, "w", newline="") as f:
            w = csv.writer(f)
            header = ["epoch", "loss"]
            if val_loss_x0 is not None: header += ["loss_x0"]
            if fid_train is not None: header += ["fid_train"]
            if fid_val is not None: header += ["fid_val"]
            # Multi-variant columns (optional)
            if fid_train_eps is not None: header += ["fid_train_eps"]
            if fid_train_x0 is not None: header += ["fid_train_x0"]
            if fid_train_combined is not None: header += ["fid_train_combined"]
            if fid_val_eps is not None: header += ["fid_val_eps"]
            if fid_val_x0 is not None: header += ["fid_val_x0"]
            if fid_val_combined is not None: header += ["fid_val_combined"]
            w.writerow(header)
            for j, ep in enumerate(val_epochs):
                row = [ep, val_loss_main[j]]
                if val_loss_x0 is not None: row += [val_loss_x0[j]]
                if fid_train is not None: row += [fid_train[j]]
                if fid_val is not None: row += [fid_val[j]]
                if fid_train_eps is not None: row += [fid_train_eps[j]]
                if fid_train_x0 is not None: row += [fid_train_x0[j]]
                if fid_train_combined is not None: row += [fid_train_combined[j]]
                if fid_val_eps is not None: row += [fid_val_eps[j]]
                if fid_val_x0 is not None: row += [fid_val_x0[j]]
                if fid_val_combined is not None: row += [fid_val_combined[j]]
                w.writerow(row)
        time.sleep(0.1)
        assert val_csv.exists() and val_csv.stat().st_size > 0, f"Val CSV not saved: {val_csv}"
        return val_csv

    # --------------- Plots ---------------
    def plot_training_main(self, train_steps: Sequence[int], train_loss_main: Sequence[float]) -> Path:
        plt.figure()
        plt.plot(train_steps, train_loss_main, label="loss (raw)")
        plt.plot(train_steps, ema_series(train_loss_main), label="loss (EMA)")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Training loss (main)"); plt.legend(); plt.tight_layout()
        pl = self.out_dir / f"{self.prefix}training_loss_main.png"
        plt.savefig(pl, dpi=150); plt.close()
        time.sleep(0.1)
        assert pl.exists() and pl.stat().st_size > 0, f"Plot not saved: {pl}"
        return pl

    def plot_training_components(
        self,
        train_steps: Sequence[int],
        train_loss_eps: Optional[Sequence[float]] = None,
        train_loss_x0: Optional[Sequence[float]] = None,
        train_loss_cons: Optional[Sequence[float]] = None,
        train_loss_total: Optional[Sequence[float]] = None,
    ) -> Optional[Path]:
        if (train_loss_eps is None) and (train_loss_x0 is None) and (train_loss_cons is None) and (train_loss_total is None):
            return None
        plt.figure()
        if train_loss_eps is not None:  plt.plot(train_steps[:len(train_loss_eps)],  ema_series(train_loss_eps),  label="loss_eps (EMA)")
        if train_loss_x0 is not None:   plt.plot(train_steps[:len(train_loss_x0)], ema_series(train_loss_x0), label="loss_x0 (EMA)")
        if train_loss_cons is not None: plt.plot(train_steps[:len(train_loss_cons)], ema_series(train_loss_cons), label="loss_cons (EMA)")
        if train_loss_total is not None:plt.plot(train_steps[:len(train_loss_total)], ema_series(train_loss_total), label="loss_total (EMA)")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Training loss components"); plt.legend(); plt.tight_layout()
        pl = self.out_dir / f"{self.prefix}training_loss_components.png"
        plt.savefig(pl, dpi=150); plt.close()
        time.sleep(0.1)
        assert pl.exists() and pl.stat().st_size > 0, f"Plot not saved: {pl}"
        return pl

    def plot_val_losses(
        self,
        val_epochs: Sequence[int],
        val_loss_main: Sequence[float],
        val_loss_x0: Optional[Sequence[float]] = None,
    ) -> Path:
        plt.figure()
        plt.plot(val_epochs, val_loss_main, label="val loss (eps)")
        if val_loss_x0 is not None: plt.plot(val_epochs, val_loss_x0, label="val loss_x0")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Validation losses"); plt.legend(); plt.tight_layout()
        pl = self.out_dir / f"{self.prefix}val_losses.png"
        plt.savefig(pl, dpi=150); plt.close()
        time.sleep(0.1)
        assert pl.exists() and pl.stat().st_size > 0, f"Plot not saved: {pl}"
        return pl

    def plot_fid(
        self,
        val_epochs: Sequence[int],
        fid_train: Optional[Sequence[float]] = None,
        fid_val: Optional[Sequence[float]] = None,
        # Optional multi-task variant FIDs
        fid_train_eps: Optional[Sequence[float]] = None,
        fid_train_x0: Optional[Sequence[float]] = None,
        fid_train_combined: Optional[Sequence[float]] = None,
        fid_val_eps: Optional[Sequence[float]] = None,
        fid_val_x0: Optional[Sequence[float]] = None,
        fid_val_combined: Optional[Sequence[float]] = None,
    ) -> Optional[Path]:
        if all(x is None for x in [fid_train, fid_val, fid_train_eps, fid_train_x0, fid_train_combined, fid_val_eps, fid_val_x0, fid_val_combined]):
            return None
        plt.figure()
        if fid_train is not None: plt.plot(val_epochs, fid_train, label="FID (train)")
        if fid_val is not None:   plt.plot(val_epochs, fid_val,   label="FID (val)")
        # Variants
        if fid_train_eps is not None: plt.plot(val_epochs, fid_train_eps, label="FID tr (eps)")
        if fid_train_x0 is not None: plt.plot(val_epochs, fid_train_x0, label="FID tr (x0)")
        if fid_train_combined is not None: plt.plot(val_epochs, fid_train_combined, label="FID tr (combined)")
        if fid_val_eps is not None: plt.plot(val_epochs, fid_val_eps, label="FID val (eps)", linestyle=":")
        if fid_val_x0 is not None: plt.plot(val_epochs, fid_val_x0, label="FID val (x0)", linestyle=":")
        if fid_val_combined is not None: plt.plot(val_epochs, fid_val_combined, label="FID val (combined)", linestyle=":")
        plt.xlabel("epoch"); plt.ylabel("FID"); plt.title("FID over epochs"); plt.legend(); plt.tight_layout()
        pl = self.out_dir / f"{self.prefix}fid.png"
        plt.savefig(pl, dpi=150); plt.close()
        time.sleep(0.1)
        assert pl.exists() and pl.stat().st_size > 0, f"Plot not saved: {pl}"
        return pl

    # --------------- Epoch-level (train vs val) ---------------
    def write_epoch_losses_csv(
        self,
        epochs: Sequence[int],
        train_loss_epoch: Sequence[float],
        val_loss_epoch: Sequence[float],
        val_loss_x0: Optional[Sequence[float]] = None,
    ) -> Path:
        csv_path = self.out_dir / f"{self.prefix}epoch_losses.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            header = ["epoch", "train_loss", "val_loss"]
            if val_loss_x0 is not None:
                header += ["val_loss_x0"]
            w.writerow(header)
            for i, ep in enumerate(epochs):
                tr = train_loss_epoch[i] if i < len(train_loss_epoch) else ""
                vl = val_loss_epoch[i] if i < len(val_loss_epoch) else ""
                row = [ep, tr, vl]
                if val_loss_x0 is not None:
                    vx = val_loss_x0[i] if i < len(val_loss_x0) else ""
                    row.append(vx)
                w.writerow(row)
        import time as _t
        _t.sleep(0.1)
        assert csv_path.exists() and csv_path.stat().st_size > 0, f"Epoch CSV not saved: {csv_path}"
        return csv_path

    def plot_epoch_train_vs_val(
        self,
        epochs: Sequence[int],
        train_loss_epoch: Sequence[float],
        val_loss_epoch: Sequence[float],
        val_loss_x0: Optional[Sequence[float]] = None,
    ) -> Path:
        plt.figure()
        plt.plot(epochs, train_loss_epoch, label="train (epoch avg)")
        plt.plot(epochs, val_loss_epoch, label="val (eps)")
        if val_loss_x0 is not None:
            plt.plot(epochs[:len(val_loss_x0)], val_loss_x0, label="val (x0)")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Loss per Epoch (train & val)"); plt.legend(); plt.tight_layout()
        pl = self.out_dir / f"{self.prefix}loss_per_epoch.png"
        plt.savefig(pl, dpi=150); plt.close()
        import time as _t
        _t.sleep(0.1)
        assert pl.exists() and pl.stat().st_size > 0, f"Plot not saved: {pl}"
        return pl

    # --------------- Orchestration ---------------
    def save_all(
        self,
        train_steps: Sequence[int],
        train_loss_main: Sequence[float],
        val_epochs: Sequence[int],
        val_loss_main: Sequence[float],
        train_loss_eps: Optional[Sequence[float]] = None,
        train_loss_x0: Optional[Sequence[float]] = None,
        train_loss_cons: Optional[Sequence[float]] = None,
        train_loss_total: Optional[Sequence[float]] = None,
        val_loss_x0: Optional[Sequence[float]] = None,
        fid_train: Optional[Sequence[float]] = None,
        fid_val: Optional[Sequence[float]] = None,
        # Optional multi-variant FIDs
        fid_train_eps: Optional[Sequence[float]] = None,
        fid_train_x0: Optional[Sequence[float]] = None,
        fid_train_combined: Optional[Sequence[float]] = None,
        fid_val_eps: Optional[Sequence[float]] = None,
        fid_val_x0: Optional[Sequence[float]] = None,
        fid_val_combined: Optional[Sequence[float]] = None,
    ) -> dict:
        # Only write CSVs; figures are generated by higher-level routines.
        train_csv = self.write_train_csv(
            train_steps, train_loss_main,
            train_loss_eps=train_loss_eps,
            train_loss_x0=train_loss_x0,
            train_loss_cons=train_loss_cons,
            train_loss_total=train_loss_total,
        )
        val_csv = self.write_val_csv(
            val_epochs, val_loss_main,
            val_loss_x0=val_loss_x0,
            fid_train=fid_train, fid_val=fid_val,
            fid_train_eps=fid_train_eps,
            fid_train_x0=fid_train_x0,
            fid_train_combined=fid_train_combined,
            fid_val_eps=fid_val_eps,
            fid_val_x0=fid_val_x0,
            fid_val_combined=fid_val_combined,
        )
        return {
            'train_csv': train_csv,
            'val_csv': val_csv,
        }
