from __future__ import annotations
from pathlib import Path
import shutil
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
# AMP (automatic mixed precision) compatibility shim: prefer torch.amp API
try:
    from torch.amp import GradScaler as _AmpGradScaler  # type: ignore
    from torch.amp import autocast as _amp_autocast      # type: ignore
    def GradScaler(*, enabled: bool = True):  # type: ignore
        return _AmpGradScaler('cuda', enabled=enabled)
    def autocast(enabled: bool = False):  # type: ignore
        return _amp_autocast('cuda', enabled=enabled)
except Exception:
    try:
        from torch.cuda.amp import GradScaler as _CudaGradScaler  # type: ignore
        from torch.cuda.amp import autocast as _cuda_autocast      # type: ignore
        def GradScaler(*, enabled: bool = True):  # type: ignore
            return _CudaGradScaler(enabled=enabled)
        def autocast(enabled: bool = False):  # type: ignore
            return _cuda_autocast(enabled=enabled)
    except Exception:
        GradScaler = None  # type: ignore
        def autocast(enabled: bool = False):  # type: ignore
            from contextlib import contextmanager
            @contextmanager
            def _noop():
                yield
            return _noop()

from models import TinyUNet, DeepSupervisedUNet
from diffusion import DDPM, DiffusionConfig
from data_utils import make_dataloader, make_test_loader
from sampling import sample
from metrics_utils import (
    denorm, dump_images, compute_fid, save_curves_unified_prefixed
)
from dataclasses import dataclass
from typing import Dict
from engine.train import train_one_epoch
from engine.validate import evaluate_mse
from losses import unified_loss as _unified_loss, deep_supervised_loss as _deep_supervised_loss


@dataclass
class TrainConfig:
    save_root: str = "runs"
    mode: str = "single"            # one of: single, multi, dsd
    data: str = "mnist"             # mnist or cifar10
    epochs: int = 1
    batch_size: int = 128
    lr: float = 2e-4
    timesteps: int = 200
    beta_start: float = 1e-4
    beta_end: float = 0.02
    base: int = 32
    time_dim: int = 128
    n_sample: int = 64
    sample_every: int = 500
    val_split: float = 0.05
    fid_eval_images: int = 1024
    w_x0: float = 1.0
    w_consistency: float = 0.1
    experiment_dir: str | Path | None = None
    multi_variant: str = "eps_x0_consistency"
    dsd_w_aux_eps: float = 0.5
    dsd_w_aux_x0: float = 0.5
    # Keep per-epoch FID image folders instead of deleting them
    keep_fid_images: bool = False


class UnifiedTrainer:
    def __init__(self, cfg: TrainConfig):
        assert cfg.mode in ["single", "multi", "dsd"]
        self.cfg = cfg
        data_l = cfg.data.lower()
        if data_l in ["cifar10", "cifar"]:
            self.ds_key = "cifar10"
        elif data_l in ["mnist", "cifar100", "svhn", "celeba"]:
            self.ds_key = data_l
        else:
            raise ValueError(f"Unsupported dataset: {cfg.data}")

        base_dir = Path(cfg.experiment_dir) if cfg.experiment_dir is not None else (Path(cfg.save_root) / "experiment")
        self.images_dir = base_dir / "images"
        self.metrics_dir = base_dir / "metrics"
        self.fid_dir = base_dir / "fid" / f"{self.ds_key}_{cfg.mode}"
        self.ckpt_dir = base_dir / "checkpoints"
        self.file_prefix = f"{self.ds_key}_{cfg.mode}"
        # Add a subfolder for visualization grids under images/
        self.grid_dir = self.images_dir / "grid"
        for d in [self.images_dir, self.grid_dir, self.metrics_dir, self.ckpt_dir]:
            d.mkdir(parents=True, exist_ok=True)
        print(f"[paths|{cfg.mode}|{self.ds_key}] images={self.images_dir} metrics={self.metrics_dir} fid_dir={self.fid_dir} ckpts={self.ckpt_dir}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.channels = 1 if self.ds_key == "mnist" else 3
        if self.ds_key == "mnist":
            self.img_size = 28
        elif self.ds_key in ["cifar10", "cifar100", "svhn"]:
            self.img_size = 32
        elif self.ds_key == "celeba":
            self.img_size = 64
        else:
            self.img_size = 32

        self.train_loader, self.val_loader, self.train_fid_loader = make_dataloader(
            self.ds_key, cfg.batch_size, self.img_size, self.channels, val_split=cfg.val_split
        )
        # Dataset sizes (print before training starts)
        try:
            train_size = len(self.train_loader.dataset)
        except Exception:
            train_size = -1
        try:
            val_size = len(self.val_loader.dataset)
        except Exception:
            val_size = -1
        try:
            _tloader = make_test_loader(self.ds_key, cfg.batch_size, self.img_size, self.channels)
            test_size = len(_tloader.dataset)
            del _tloader
        except Exception:
            test_size = -1
        _ts = test_size if isinstance(test_size, int) and test_size >= 0 else 'unknown'
        print(f"[data|{cfg.mode}|{self.ds_key}] train={train_size} val={val_size} test={_ts} batch={cfg.batch_size}")

        if cfg.mode == "dsd":
            self.model: torch.nn.Module = DeepSupervisedUNet(in_channels=self.channels, base=cfg.base, time_dim=cfg.time_dim).to(self.device)
        else:
            self.model = TinyUNet(in_channels=self.channels, base=cfg.base, time_dim=cfg.time_dim, multi_task=(cfg.mode=="multi")).to(self.device)
        self.ddpm = DDPM(DiffusionConfig(timesteps=cfg.timesteps, beta_start=cfg.beta_start, beta_end=cfg.beta_end)).to(self.device)
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)
        # AMP scaler for faster training on CUDA
        self.use_amp = bool(torch.cuda.is_available()) and (GradScaler is not None)
        self.scaler = GradScaler(enabled=self.use_amp) if GradScaler is not None else None

        # History containers
        self.step = 0
        # Per-epoch training averages for easier comparison with validation
        self.tr_epoch_loss_main: list[float] = []
        self.tr_epoch_loss_x0: list[float] = []
        self.tr_epoch_loss_cons: list[float] = []
        self.tr_epoch_loss_total: list[float] = []
        self.va_loss_main: list[float] = []
        self.va_loss_x0: list[float] = []
        self.fid_train_hist: list[float] = []
        self.fid_val_hist: list[float] = []
        self.best_val_loss = float('inf')
        self.best_val_fid = float('inf')

    def train_epoch(self, epoch_idx: int):
        return train_one_epoch(self, epoch_idx)

    @torch.no_grad()
    def evaluate_mse(self) -> Dict[str, float]:
        return evaluate_mse(self.model, self.ddpm, self.val_loader, self.device, self.cfg.mode == "multi")

    def _make_epoch_fid_dirs(self, epoch_idx: int):
        """Create per-epoch FID directories and return them.
        Layout under fid/{file_prefix}/epoch_{k}/: train_real, train_fake, val_real, val_fake
        """
        ep_dir = self.fid_dir / f"epoch_{epoch_idx+1}"
        tr_real = ep_dir / "train_real"
        tr_fake = ep_dir / "train_fake"
        va_real = ep_dir / "val_real"
        va_fake = ep_dir / "val_fake"
        for d in [tr_real, tr_fake, va_real, va_fake]:
            d.mkdir(parents=True, exist_ok=True)
        return {
            'epoch_dir': ep_dir,
            'train_real': tr_real,
            'train_fake': tr_fake,
            'val_real': va_real,
            'val_fake': va_fake,
        }

    @torch.no_grad()
    def _collect_train_subset(self, epoch_idx: int, fid_dirs):
        """Prepare a subset of the training set for FID using cfg.fid_eval_images.

        Writes PNGs into train_real and train_fake under the epoch folder.
        """
        # Real subset
        collected = 0
        imgs_accum = []
        for imgs, _ in self.train_fid_loader:
            imgs_accum.append(imgs)
            collected += imgs.size(0)
            if collected >= self.cfg.fid_eval_images:
                break
        real_train = torch.cat(imgs_accum, dim=0)[: self.cfg.fid_eval_images]
        real_train = torch.nan_to_num(denorm(real_train, self.channels), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        dump_images(real_train, str(fid_dirs['train_real']), prefix=f"{self.file_prefix}_real")

        # Fake subset (generate in chunks)
        written = 0
        while written < self.cfg.fid_eval_images:
            n = min(self.cfg.n_sample, self.cfg.fid_eval_images - written)
            _, fb = sample(
                self.model,
                self.ddpm,
                shape=(n, self.channels, self.img_size, self.img_size),
                device=self.device,
                save_path=str(self.grid_dir / f"{self.file_prefix}_grid_trainfid_epoch{epoch_idx+1}.png"),
            )
            batch = torch.nan_to_num(denorm(fb.cpu(), self.channels), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            dump_images(batch, str(fid_dirs['train_fake']), prefix=f"{self.file_prefix}_fake", start_index=written)
            written += batch.size(0)

    @torch.no_grad()
    def _collect_val_subset(self, epoch_idx: int, fid_dirs):
        """Prepare a subset of the validation set for FID using cfg.fid_eval_images.

        Writes PNGs into val_real and val_fake under the epoch folder.
        """
        # Real subset from validation loader
        collected = 0
        imgs_accum = []
        for imgs, _ in self.val_loader:
            imgs_accum.append(imgs)
            collected += imgs.size(0)
            if collected >= self.cfg.fid_eval_images:
                break
        real_val = torch.cat(imgs_accum, dim=0)[: self.cfg.fid_eval_images]
        real_val = torch.nan_to_num(denorm(real_val, self.channels), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        dump_images(real_val, str(fid_dirs['val_real']), prefix=f"{self.file_prefix}_real")

        # Fake subset (generate in chunks)
        written = 0
        while written < self.cfg.fid_eval_images:
            n = min(self.cfg.n_sample, self.cfg.fid_eval_images - written)
            _, fb = sample(
                self.model,
                self.ddpm,
                shape=(n, self.channels, self.img_size, self.img_size),
                device=self.device,
                save_path=str(self.grid_dir / f"{self.file_prefix}_grid_valfid_epoch{epoch_idx+1}.png"),
            )
            batch = torch.nan_to_num(denorm(fb.cpu(), self.channels), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            dump_images(batch, str(fid_dirs['val_fake']), prefix=f"{self.file_prefix}_fake", start_index=written)
            written += batch.size(0)

    def _collect_full_sets(self, epoch_idx: int, fid_dirs):
        # Train real (stream)
        idx = 0
        for imgs, _ in self.train_fid_loader:
            batch = torch.nan_to_num(denorm(imgs, self.channels), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            dump_images(batch, str(fid_dirs['train_real']), prefix=f"{self.file_prefix}_real", start_index=idx)
            idx += batch.size(0)
        # Val real (stream)
        idx = 0
        for imgs, _ in self.val_loader:
            batch = torch.nan_to_num(denorm(imgs, self.channels), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            dump_images(batch, str(fid_dirs['val_real']), prefix=f"{self.file_prefix}_real", start_index=idx)
            idx += batch.size(0)
        # Train fake to match train size
        train_target = len(self.train_fid_loader.dataset)
        written = 0
        while written < train_target:
            n = min(self.cfg.batch_size, train_target - written)
            _, fb = sample(self.model, self.ddpm,
                          shape=(n, self.channels, self.img_size, self.img_size),
                          device=self.device,
                          save_path=str(self.grid_dir / f"{self.file_prefix}_grid_trainfid_epoch{epoch_idx+1}.png"))
            batch = torch.nan_to_num(denorm(fb.cpu(), self.channels), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            dump_images(batch, str(fid_dirs['train_fake']), prefix=f"{self.file_prefix}_fake", start_index=written)
            written += batch.size(0)
        # Val fake to match val size
        val_target = len(self.val_loader.dataset)
        written = 0
        while written < val_target:
            n = min(self.cfg.batch_size, val_target - written)
            _, fbv = sample(self.model, self.ddpm,
                            shape=(n, self.channels, self.img_size, self.img_size),
                            device=self.device,
                            save_path=str(self.grid_dir / f"{self.file_prefix}_grid_valfid_epoch{epoch_idx+1}.png"))
            batch = torch.nan_to_num(denorm(fbv.cpu(), self.channels), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            dump_images(batch, str(fid_dirs['val_fake']), prefix=f"{self.file_prefix}_fake", start_index=written)
            written += batch.size(0)

    def _save_checkpoints(self, epoch_val_loss: float, fid_val: float):
        if epoch_val_loss < self.best_val_loss:
            self.best_val_loss = epoch_val_loss
            _ckpt_loss = self.ckpt_dir / f"{self.file_prefix}_best_loss.pt"
            torch.save({
                'state_dict': self.model.state_dict(),
                'mode': self.cfg.mode,
                'ds_key': self.ds_key,
                'base': self.cfg.base,
                'time_dim': self.cfg.time_dim,
                'channels': self.channels,
                'timesteps': self.cfg.timesteps,
                'beta_start': self.cfg.beta_start,
                'beta_end': self.cfg.beta_end,
            }, _ckpt_loss)
            assert _ckpt_loss.exists() and _ckpt_loss.stat().st_size > 0, f"Checkpoint not saved: {_ckpt_loss}"
        if float(fid_val) < self.best_val_fid:
            self.best_val_fid = float(fid_val)
            _ckpt_fid = self.ckpt_dir / f"{self.file_prefix}_best_fid.pt"
            torch.save({
                'state_dict': self.model.state_dict(),
                'mode': self.cfg.mode,
                'ds_key': self.ds_key,
                'base': self.cfg.base,
                'time_dim': self.cfg.time_dim,
                'channels': self.channels,
                'timesteps': self.cfg.timesteps,
                'beta_start': self.cfg.beta_start,
                'beta_end': self.cfg.beta_end,
            }, _ckpt_fid)
            assert _ckpt_fid.exists() and _ckpt_fid.stat().st_size > 0, f"Checkpoint not saved: {_ckpt_fid}"

    def _save_metrics(self):
        # Persist per-epoch metrics to CSVs (train per-epoch, not per-step)
        epochs = list(range(1, len(self.tr_epoch_loss_main)+1))
        save_curves_unified_prefixed(
            self.metrics_dir,
            train_steps=epochs,
            train_loss_main=self.tr_epoch_loss_main,
            val_epochs=list(range(1, len(self.va_loss_main)+1)),
            val_loss_main=self.va_loss_main,
            train_loss_eps=None,
            train_loss_x0=(self.tr_epoch_loss_x0 if self.cfg.mode=="multi" else None),
            train_loss_cons=(self.tr_epoch_loss_cons if self.cfg.mode=="multi" else None),
            train_loss_total=(self.tr_epoch_loss_total if self.cfg.mode=="multi" else None),
            val_loss_x0=(self.va_loss_x0 if self.cfg.mode=="multi" else None),
            fid_train=self.fid_train_hist,
            fid_val=self.fid_val_hist,
            file_prefix=self.file_prefix,
        )
        # Additionally, persist per-epoch training vs validation losses
        try:
            from metrics.metrics_logger import MetricsLogger  # type: ignore
            import time
            logger = MetricsLogger(self.metrics_dir, self.file_prefix)
            epochs = list(range(1, len(self.tr_epoch_loss_main)+1))
            epoch_csv = logger.write_epoch_losses_csv(
                epochs,
                train_loss_epoch=self.tr_epoch_loss_main,
                val_loss_epoch=self.va_loss_main,
                val_loss_x0=(self.va_loss_x0 if self.cfg.mode=="multi" else None),
            )
            time.sleep(0.1)
            assert epoch_csv.exists() and epoch_csv.stat().st_size > 0, f"Epoch loss CSV not saved: {epoch_csv}"
            epoch_plot = logger.plot_epoch_train_vs_val(
                epochs,
                train_loss_epoch=self.tr_epoch_loss_main,
                val_loss_epoch=self.va_loss_main,
                val_loss_x0=(self.va_loss_x0 if self.cfg.mode=="multi" else None),
            )
            time.sleep(0.1)
            assert epoch_plot.exists() and epoch_plot.stat().st_size > 0, f"Epoch loss plot not saved: {epoch_plot}"
        except Exception as e:
            print(f"[warn|metrics-epoch] failed to save epoch-level metrics: {e}")

    def run(self):
        for epoch in range(self.cfg.epochs):
            avg = self.train_epoch(epoch)
            # Record per-epoch averages
            self.tr_epoch_loss_main.append(float(avg.get('loss', float('nan'))))
            if self.cfg.mode in ("multi", "dsd"):
                self.tr_epoch_loss_x0.append(float(avg.get('loss_x0', float('nan'))))
                self.tr_epoch_loss_cons.append(float(avg.get('loss_cons', float('nan'))))
                self.tr_epoch_loss_total.append(float(avg.get('loss_total', float('nan'))))

            # For multi-task, log a grid once per epoch (not per step)
            if self.cfg.mode == "multi":
                try:
                    with torch.no_grad():
                        grid_path = self.grid_dir / f"{self.file_prefix}_samples_epoch{epoch+1}.png"
                        sample(self.model, self.ddpm,
                               shape=(self.cfg.n_sample, self.channels, self.img_size, self.img_size),
                               device=self.device, save_path=str(grid_path))
                except Exception as e:
                    print(f"[warn|sample-grid-epoch|multi|{self.ds_key}] epoch={epoch+1} failed: {e}")

            val_metrics = self.evaluate_mse()
            self.va_loss_main.append(float(val_metrics["loss"]))
            if self.cfg.mode == "multi":
                self.va_loss_x0.append(float(val_metrics["loss_x0"]))

            # Prepare sets and compute FID (training subset only; test handled post-training)
            try:
                fid_dirs = self._make_epoch_fid_dirs(epoch)
                # Training FID subset
                self._collect_train_subset(epoch, fid_dirs)
                fid_train = compute_fid(
                    str(fid_dirs['train_real']), str(fid_dirs['train_fake']), self.device, fid_batch_size=self.cfg.batch_size
                )
                # Validation FID subset
                self._collect_val_subset(epoch, fid_dirs)
                fid_val = compute_fid(
                    str(fid_dirs['val_real']), str(fid_dirs['val_fake']), self.device, fid_batch_size=self.cfg.batch_size
                )
                # Optionally clean up whole epoch dir after both computations
                if not self.cfg.keep_fid_images:
                    try:
                        shutil.rmtree(fid_dirs.get('epoch_dir', Path("")), ignore_errors=True)
                    except Exception:
                        pass
                else:
                    print(f"[fid] kept epoch {epoch+1} FID images at {fid_dirs['epoch_dir']}")
            except Exception as e:
                print(f"[warn|fid|{self.cfg.mode}|{self.ds_key}] epoch={epoch+1} FID pipeline failed: {e}")
                fid_train = float('nan')
                fid_val = float('nan')
            self.fid_train_hist.append(float(fid_train))
            self.fid_val_hist.append(float(fid_val))

            # Save best checkpoints (track best by validation loss; best FID checkpoint based on test is handled after testing if desired)
            self._save_checkpoints(self.va_loss_main[-1], float('inf'))

            # Validation log summary
            if self.cfg.mode == "multi":
                print(f"[val] epoch {epoch+1}: loss={self.va_loss_main[-1]:.6f} | loss_x0={self.va_loss_x0[-1]:.6f} | FID_train={fid_train:.2f} | FID_val={fid_val:.2f}")
            else:
                print(f"[val] epoch {epoch+1}: loss={self.va_loss_main[-1]:.6f} | FID_train={fid_train:.2f} | FID_val={fid_val:.2f}")

            # Persist metrics/plots for this epoch
            self._save_metrics()

        # Final sampling and metrics persistence
        with torch.no_grad():
            path, _ = sample(
                self.model, self.ddpm,
                shape=(self.cfg.n_sample, self.channels, self.img_size, self.img_size),
                device=self.device, save_path=str(self.grid_dir / f"{self.file_prefix}_samples_final.png"),
            )
            p = Path(path)
            assert p.exists() and p.stat().st_size > 0, f"Final sample not saved: {path}"
        print(f"Saved final samples to {path}")

        self._save_metrics()

        # Proactively release resources
        try:
            del self.train_loader, self.val_loader, self.train_fid_loader
        except Exception:
            pass
        try:
            del self.model, self.ddpm, self.optim
        except Exception:
            pass
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


def unified_loss(model, ddpm: DDPM, x0, t, multi_task: bool, w_x0=1.0, w_consistency=0.1, multi_variant: str = "eps_x0_consistency"):
    # Delegate to losses module for a single source of truth
    return _unified_loss(model, ddpm, x0, t, multi_task, w_x0, w_consistency, multi_variant)


def deep_supervised_loss(model: DeepSupervisedUNet, ddpm: DDPM, x0, t,
                         w_aux_eps: float = 0.5, w_aux_x0: float = 0.5):
    # Delegate to losses module for a single source of truth
    return _deep_supervised_loss(model, ddpm, x0, t, w_aux_eps, w_aux_x0)


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
    keep_fid_images: bool = False,
):
    cfg = TrainConfig(
        save_root=save_root,
        mode=mode,
        data=data,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        timesteps=timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        base=base,
        time_dim=time_dim,
        n_sample=n_sample,
        sample_every=sample_every,
        val_split=val_split,
        fid_eval_images=fid_eval_images,
        w_x0=w_x0,
        w_consistency=w_consistency,
        experiment_dir=experiment_dir,
        multi_variant=multi_variant,
        dsd_w_aux_eps=dsd_w_aux_eps,
        dsd_w_aux_x0=dsd_w_aux_x0,
        keep_fid_images=keep_fid_images,
    )
    UnifiedTrainer(cfg).run()


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
