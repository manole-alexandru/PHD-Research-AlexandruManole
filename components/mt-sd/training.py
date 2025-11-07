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


class UnifiedTrainer:
    def __init__(self, cfg: TrainConfig):
        assert cfg.mode in ["single", "multi", "dsd"]
        self.cfg = cfg
        self.ds_key = "cifar10" if cfg.data.lower() in ["cifar10", "cifar"] else "mnist"

        base_dir = Path(cfg.experiment_dir) if cfg.experiment_dir is not None else (Path(cfg.save_root) / "experiment")
        self.images_dir = base_dir / "images"
        self.metrics_dir = base_dir / "metrics"
        self.fid_dir = base_dir / "fid" / f"{self.ds_key}_{cfg.mode}"
        self.ckpt_dir = base_dir / "checkpoints"
        self.fid_train_real = self.fid_dir / "train_real"
        self.fid_train_fake = self.fid_dir / "train_fake"
        self.fid_val_real = self.fid_dir / "val_real"
        self.fid_val_fake = self.fid_dir / "val_fake"
        self.file_prefix = f"{self.ds_key}_{cfg.mode}"
        for d in [self.images_dir, self.metrics_dir, self.fid_train_real, self.fid_train_fake, self.fid_val_real, self.fid_val_fake, self.ckpt_dir]:
            d.mkdir(parents=True, exist_ok=True)
        print(f"[paths|{cfg.mode}|{self.ds_key}] images={self.images_dir} metrics={self.metrics_dir} fid_dir={self.fid_dir} ckpts={self.ckpt_dir}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.channels = 1 if self.ds_key == "mnist" else 3
        self.img_size = 28 if self.ds_key == "mnist" else 32

        self.train_loader, self.val_loader, self.train_fid_loader = make_dataloader(
            self.ds_key, cfg.batch_size, self.img_size, self.channels, val_split=cfg.val_split
        )

        if cfg.mode == "dsd":
            self.model: torch.nn.Module = DeepSupervisedUNet(in_channels=self.channels, base=cfg.base, time_dim=cfg.time_dim).to(self.device)
        else:
            self.model = TinyUNet(in_channels=self.channels, base=cfg.base, time_dim=cfg.time_dim, multi_task=(cfg.mode=="multi")).to(self.device)
        self.ddpm = DDPM(DiffusionConfig(timesteps=cfg.timesteps, beta_start=cfg.beta_start, beta_end=cfg.beta_end)).to(self.device)
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)

        # History containers
        self.step = 0
        self.train_steps = []
        self.tr_loss_main: list[float] = []
        self.tr_loss_x0: list[float] = []
        self.tr_loss_cons: list[float] = []
        self.tr_loss_total: list[float] = []
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

    def _prepare_fid_dirs(self):
        # Clean previous images for this prefix so FID computes on current epoch only
        for d in [self.fid_train_real, self.fid_train_fake, self.fid_val_real, self.fid_val_fake]:
            for f in d.glob(f"{self.file_prefix}_*.png"):
                try:
                    f.unlink()
                except Exception:
                    pass

    def _collect_real_sets(self):
        # Train real
        collected = 0; imgs_accum = []
        for imgs, _ in self.train_fid_loader:
            imgs_accum.append(imgs)
            collected += imgs.size(0)
            if collected >= self.cfg.fid_eval_images: break
        real_train = torch.cat(imgs_accum, dim=0)[:self.cfg.fid_eval_images]
        real_train = denorm(real_train, self.channels)
        dump_images(real_train, str(self.fid_train_real), prefix=f"{self.file_prefix}_real")
        # Val real
        collected = 0; imgs_accum = []
        for imgs, _ in self.val_loader:
            imgs_accum.append(imgs)
            collected += imgs.size(0)
            if collected >= self.cfg.fid_eval_images: break
        real_val = torch.cat(imgs_accum, dim=0)[:self.cfg.fid_eval_images]
        real_val = denorm(real_val, self.channels)
        dump_images(real_val, str(self.fid_val_real), prefix=f"{self.file_prefix}_real")

    @torch.no_grad()
    def _collect_fake_sets(self, epoch_idx: int):
        _, fake_batch = sample(
            self.model, self.ddpm,
            shape=(min(self.cfg.fid_eval_images, self.cfg.n_sample), self.channels, self.img_size, self.img_size),
            device=self.device, save_path=str(self.images_dir / f"{self.file_prefix}_samples_trainfid_epoch{epoch_idx+1}.png"),
        )
        fake_list = [denorm(fake_batch.cpu(), self.channels)]
        while sum(x.size(0) for x in fake_list) < self.cfg.fid_eval_images:
            _, fb = sample(
                self.model, self.ddpm,
                shape=(min(self.cfg.fid_eval_images - sum(x.size(0) for x in fake_list), self.cfg.n_sample),
                       self.channels, self.img_size, self.img_size),
                device=self.device, save_path=str(self.images_dir / "_tmp.png"),
            )
            fake_list.append(denorm(fb.cpu(), self.channels))
        fake_train = torch.cat(fake_list, dim=0)[:self.cfg.fid_eval_images]
        dump_images(fake_train, str(self.fid_train_fake), prefix=f"{self.file_prefix}_fake")

        _, fake_batch_val = sample(
            self.model, self.ddpm,
            shape=(min(self.cfg.fid_eval_images, self.cfg.n_sample), self.channels, self.img_size, self.img_size),
            device=self.device, save_path=str(self.images_dir / f"{self.file_prefix}_samples_valfid_epoch{epoch_idx+1}.png"),
        )
        fake_list_val = [denorm(fake_batch_val.cpu(), self.channels)]
        while sum(x.size(0) for x in fake_list_val) < self.cfg.fid_eval_images:
            _, fb = sample(
                self.model, self.ddpm,
                shape=(min(self.cfg.fid_eval_images - sum(x.size(0) for x in fake_list_val), self.cfg.n_sample),
                       self.channels, self.img_size, self.img_size),
                device=self.device, save_path=str(self.images_dir / "_tmp2.png"),
            )
            fake_list_val.append(denorm(fb.cpu(), self.channels))
        fake_val = torch.cat(fake_list_val, dim=0)[:self.cfg.fid_eval_images]
        dump_images(fake_val, str(self.fid_val_fake), prefix=f"{self.file_prefix}_fake")

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
        save_curves_unified_prefixed(
            self.metrics_dir,
            train_steps=self.train_steps,
            train_loss_main=self.tr_loss_main,
            val_epochs=list(range(1, len(self.va_loss_main)+1)),
            val_loss_main=self.va_loss_main,
            train_loss_eps=(self.tr_loss_main if (self.cfg.mode=="multi") else None),
            train_loss_x0=(self.tr_loss_x0 if self.cfg.mode=="multi" else None),
            train_loss_cons=(self.tr_loss_cons if self.cfg.mode=="multi" else None),
            train_loss_total=(self.tr_loss_total if self.cfg.mode=="multi" else None),
            val_loss_x0=(self.va_loss_x0 if self.cfg.mode=="multi" else None),
            fid_train=self.fid_train_hist,
            fid_val=self.fid_val_hist,
            file_prefix=self.file_prefix,
        )

    def run(self):
        for epoch in range(self.cfg.epochs):
            self.train_epoch(epoch)

            val_metrics = self.evaluate_mse()
            self.va_loss_main.append(float(val_metrics["loss"]))
            if self.cfg.mode == "multi":
                self.va_loss_x0.append(float(val_metrics["loss_x0"]))

            # Prepare sets and compute FID
            try:
                self._prepare_fid_dirs()
                self._collect_real_sets()
                self._collect_fake_sets(epoch)
                fid_train = compute_fid(str(self.fid_train_real), str(self.fid_train_fake), self.device)
                fid_val   = compute_fid(str(self.fid_val_real),   str(self.fid_val_fake),   self.device)
            except Exception as e:
                print(f"[warn|fid|{self.cfg.mode}|{self.ds_key}] epoch={epoch+1} FID pipeline failed: {e}")
                fid_train, fid_val = float('nan'), float('nan')
            self.fid_train_hist.append(float(fid_train))
            self.fid_val_hist.append(float(fid_val))

            # Save best checkpoints
            self._save_checkpoints(self.va_loss_main[-1], fid_val)

            # Validation log summary
            if self.cfg.mode == "multi":
                print(f"[val] epoch {epoch+1}: loss={self.va_loss_main[-1]:.6f} | loss_x0={self.va_loss_x0[-1]:.6f} | "
                      f"FID_train={fid_train:.2f} | FID_val={fid_val:.2f}")
            else:
                print(f"[val] epoch {epoch+1}: loss={self.va_loss_main[-1]:.6f} | "
                      f"FID_train={fid_train:.2f} | FID_val={fid_val:.2f}")

            # Persist metrics/plots for this epoch
            self._save_metrics()

        # Final sampling and metrics persistence
        with torch.no_grad():
            path, _ = sample(
                self.model, self.ddpm,
                shape=(self.cfg.n_sample, self.channels, self.img_size, self.img_size),
                device=self.device, save_path=str(self.images_dir / f"{self.file_prefix}_samples_final.png"),
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
