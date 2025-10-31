# Tiny Diffusion (DDPM-style) — unified single/multi-task training
# Colab-friendly, one cell; everything saved under save_root/<mode>/<dataset>/

from __future__ import annotations
import math, os, csv
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils
from tqdm import tqdm

# Headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------------
# Time embedding utilities
# -------------------------
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    def forward(self, t: torch.Tensor):
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = None
        if time_dim is not None:
            self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.act = nn.SiLU()
        self.norm1 = nn.GroupNorm(4, out_ch)
        self.norm2 = nn.GroupNorm(4, out_ch)
    def forward(self, x, t_emb=None):
        x = self.conv1(x)
        if self.time_mlp is not None and t_emb is not None:
            x = x + self.time_mlp(t_emb)[:, :, None, None]
        x = self.norm1(x); x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x); x = self.act(x)
        return x

class TinyUNet(nn.Module):
    """
    One model that supports:
      - single-task: returns Tensor (predicted ε)
      - multi-task : returns dict {'eps': ε_hat, 'x0': x0_hat}
    """
    def __init__(self, in_channels=1, base=32, time_dim=128, multi_task: bool=False):
        super().__init__()
        self.multi_task = multi_task
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )
        self.inc = ConvBlock(in_channels, base, time_dim)
        self.down1 = nn.Sequential(nn.Conv2d(base, base, 3, stride=2, padding=1), nn.SiLU())
        self.block1 = ConvBlock(base, base * 2, time_dim)
        self.down2 = nn.Sequential(nn.Conv2d(base * 2, base * 2, 3, stride=2, padding=1), nn.SiLU())
        self.block2 = ConvBlock(base * 2, base * 4, time_dim)
        self.mid = ConvBlock(base * 4, base * 4, time_dim)
        self.up1 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.block_up1 = ConvBlock(base * 4, base * 2, time_dim)
        self.up2 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.block_up2 = ConvBlock(base * 2, base, time_dim)

        # Heads
        self.out_eps = nn.Conv2d(base, in_channels, 1)
        if self.multi_task:
            self.out_x0 = nn.Conv2d(base, in_channels, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x0 = self.inc(x, t_emb)
        x1 = self.down1(x0); x1 = self.block1(x1, t_emb)
        x2 = self.down2(x1); x2 = self.block2(x2, t_emb)
        m = self.mid(x2, t_emb)
        u1 = self.up1(m); u1 = torch.cat([u1, x1], dim=1); u1 = self.block_up1(u1, t_emb)
        u2 = self.up2(u1); u2 = torch.cat([u2, x0], dim=1); u2 = self.block_up2(u2, t_emb)
        if self.multi_task:
            return {"eps": self.out_eps(u2), "x0": self.out_x0(u2)}
        else:
            return self.out_eps(u2)  # ε only

@dataclass
class DiffusionConfig:
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02

class DDPM(nn.Module):
    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        self.cfg = cfg
        self._register_buffers()

    def _register_buffers(self):
        T = self.cfg.timesteps
        betas = torch.linspace(self.cfg.beta_start, self.cfg.beta_end, T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # register as buffers so .to(device) works
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",
                             torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("posterior_variance",
                             betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_om = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_ac * x0 + sqrt_om * noise, noise

    @staticmethod
    def _extract(a, t, x_shape):
        # a and t are on the same device because 'a' is a registered buffer
        b = t.shape[0]
        out = a.gather(0, t).float().view(b, *((1,) * (len(x_shape) - 1)))
        return out

# ---- Data ----
def make_dataloader(name: str, batch_size: int, img_size: int, channels: int, val_split: float = 0.05):
    tfm = [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
    if channels == 3:
        tfm = [transforms.Resize(img_size), transforms.ToTensor(),
               transforms.Normalize((0.5,)*3, (0.5,)*3)]
    tfm = transforms.Compose(tfm)
    if name.lower() == "mnist":
        ds_train = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
        ds_full_for_train_fid = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
        ds_val_src = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    elif name.lower() in ["cifar10", "cifar"]:
        name = "cifar10"
        ds_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm)
        ds_full_for_train_fid = datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm)
        ds_val_src = datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm)
    else:
        raise ValueError("Unsupported dataset. Use 'mnist' or 'cifar10'.")

    # Split train into train/val
    val_size = max(1, int(len(ds_train) * val_split))
    train_size = len(ds_train) - val_size
    train_ds, val_ds = torch.utils.data.random_split(ds_train, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # A loader over the original train split (for "train FID" real images)
    train_fid_loader = DataLoader(ds_full_for_train_fid, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, train_fid_loader

# ---- Sampling (always uses ε) ----
@torch.no_grad()
def sample(model, ddpm: DDPM, shape, device, save_path="samples.png"):
    model.eval()
    T = ddpm.cfg.timesteps
    x = torch.randn(shape, device=device)
    for i in tqdm(reversed(range(T)), total=T, desc="sampling"):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        beta_t = ddpm._extract(ddpm.betas, t, x.shape)
        sqrt_one_minus_ac = ddpm._extract(ddpm.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alpha = ddpm._extract(ddpm.sqrt_recip_alphas, t, x.shape)
        preds = model(x, t.float())
        pred_eps = preds["eps"] if isinstance(preds, dict) else preds
        x0_pred = (x - sqrt_one_minus_ac * pred_eps) * sqrt_recip_alpha
        alphas = ddpm._extract(ddpm.alphas, t, x.shape)
        alphas_cum = ddpm._extract(ddpm.alphas_cumprod, t, x.shape)
        alphas_cum_prev = ddpm._extract(ddpm.alphas_cumprod_prev, t, x.shape)
        posterior_var = ddpm._extract(ddpm.posterior_variance, t, x.shape)
        posterior_mean = (
            (beta_t * torch.sqrt(alphas_cum_prev) / (1.0 - alphas_cum)) * x0_pred
            + ((torch.sqrt(alphas) * (1.0 - alphas_cum_prev)) / (1.0 - alphas_cum)) * x
        )
        noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
        x = posterior_mean + torch.sqrt(posterior_var) * noise
    grid = vutils.make_grid((x.clamp(-1, 1) + 1) * 0.5, nrow=int(math.sqrt(shape[0])))
    vutils.save_image(grid, save_path)
    return save_path, x.detach().cpu()

# ---- Validation utils ----
def denorm(x, channels):
    x = (x.clamp(-1, 1) + 1) * 0.5
    if channels == 1:
        x = x.repeat(1, 3, 1, 1)  # Inception expects 3 channels
    return x

@torch.no_grad()
def evaluate_mse_unified(model, ddpm: DDPM, data_loader, device, multi_task: bool):
    """
    Returns dict with:
      - 'loss'           : main ε MSE averaged over pixels (matches single-task naming)
      - 'loss_x0'        : avg x0 MSE (multi-task only, else None)
    """
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
            eps = preds
            mse_eps = F.mse_loss(eps, noise, reduction='sum')
            tot_eps += mse_eps.item()
        denom += imgs.numel()
    denom = max(1, denom)
    out = {"loss": tot_eps / denom}
    out["loss_x0"] = (tot_x0 / denom) if multi_task else None
    return out

@torch.no_grad()
def dump_images(tensor_bchw, out_dir: str, prefix: str = "img"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(tensor_bchw):
        vutils.save_image(img, out / f"{prefix}_{i:05d}.png")

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

# ---- Helpers: EMA & plots/CSVs ----
def ema_series(values, decay=0.98):
    if not values: return []
    out, m = [], values[0]
    for v in values:
        m = decay * m + (1 - decay) * v
        out.append(m)
    return out

def save_curves_unified(
    out_dir: Path,
    train_steps,  # ints
    # main losses (always present)
    train_loss_main, val_epochs, val_loss_main,
    # optional components (multi-task)
    train_loss_x0=None, train_loss_cons=None, train_loss_total=None,
    val_loss_x0=None,
    # FIDs
    fid_train=None, fid_val=None,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV: training losses
    with open(out_dir / "train_losses.csv", "w", newline="") as f:
        w = csv.writer(f)
        header = ["step", "loss"]  # 'loss' = main ε objective (consistent across modes)
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

    # CSV: validation metrics
    with open(out_dir / "val_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        header = ["epoch", "loss"]  # 'loss' = main ε objective
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

    # Plots
    # 1) Training main loss
    plt.figure()
    plt.plot(train_steps, train_loss_main, label="loss (raw)")
    plt.plot(train_steps, ema_series(train_loss_main), label="loss (EMA)")
    plt.xlabel("step"); plt.ylabel("loss"); plt.title("Training loss (main ε)"); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "training_loss_main.png", dpi=150); plt.close()

    # 2) (Optional) Training components
    if train_loss_x0 is not None or train_loss_cons is not None or train_loss_total is not None:
        plt.figure()
        if train_loss_x0 is not None:   plt.plot(train_steps[:len(train_loss_x0)], ema_series(train_loss_x0), label="loss_x0 (EMA)")
        if train_loss_cons is not None: plt.plot(train_steps[:len(train_loss_cons)], ema_series(train_loss_cons), label="loss_cons (EMA)")
        if train_loss_total is not None:plt.plot(train_steps[:len(train_loss_total)], ema_series(train_loss_total), label="loss_total (EMA)")
        plt.xlabel("step"); plt.ylabel("loss"); plt.title("Training loss components"); plt.legend(); plt.tight_layout()
        plt.savefig(out_dir / "training_loss_components.png", dpi=150); plt.close()

    # 3) Validation losses + FID
    plt.figure()
    plt.plot(val_epochs, val_loss_main, label="val loss (ε)")
    if val_loss_x0 is not None: plt.plot(val_epochs, val_loss_x0, label="val loss_x0")
    if fid_train is not None:   plt.plot(val_epochs, fid_train, label="FID (train)")
    if fid_val is not None:     plt.plot(val_epochs, fid_val, label="FID (val)")
    plt.xlabel("epoch"); plt.ylabel("metric"); plt.title("Validation metrics"); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "val_metrics.png", dpi=150); plt.close()

# ---- Unified loss (single or multi) ----
def unified_loss(model, ddpm: DDPM, x0, t, multi_task: bool, w_x0=1.0, w_consistency=0.1):
    """
    Returns:
      total_loss, parts dict with keys:
        - 'loss'       : main ε loss (ALWAYS present, consistent name)
        - 'loss_x0'    : (multi only)
        - 'loss_cons'  : (multi only)
        - 'loss_total' : (multi only)
    """
    x_t, noise = ddpm.q_sample(x0, t)
    preds = model(x_t, t.float())
    if not multi_task:
        loss_eps = F.mse_loss(preds, noise)
        return loss_eps, {"loss": loss_eps.detach()}
    # multi-task
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

    total = loss_eps + w_x0 * loss_x0 + w_consistency * loss_cons
    return total, {
        "loss": loss_eps.detach(),        # main objective name matches single-task
        "loss_x0": loss_x0.detach(),
        "loss_cons": loss_cons.detach(),
        "loss_total": total.detach()
    }

# ---- Unified trainer ----
def train_unified(
    save_root: str = "runs",
    mode: str = "single",               # 'single' or 'multi'
    data: str = "mnist",                # 'mnist' or 'cifar10'
    epochs: int = 1,
    batch_size: int = 128,
    lr: float = 2e-4,
    timesteps: int = 200,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    base: int = 32,
    time_dim: int = 128,
    n_sample: int = 64,                 # samples per grid
    sample_every: int = 500,            # steps
    val_split: float = 0.05,
    fid_eval_images: int = 1024,        # FID pool size
    w_x0: float = 1.0,                  # only used in 'multi'
    w_consistency: float = 0.1,         # only used in 'multi'
):
    assert mode in ["single", "multi"]
    ds_key = "cifar10" if data.lower() in ["cifar10", "cifar"] else "mnist"

    # --- Paths ---
    save_dir = Path(save_root) / mode / ds_key
    images_dir = save_dir / "images"
    metrics_dir = save_dir / "metrics"
    fid_dir = save_dir / "fid"
    fid_train_real = fid_dir / "train_real"
    fid_train_fake = fid_dir / "train_fake"
    fid_val_real   = fid_dir / "val_real"
    fid_val_fake   = fid_dir / "val_fake"
    for d in [images_dir, metrics_dir, fid_train_real, fid_train_fake, fid_val_real, fid_val_fake]:
        d.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channels = 1 if ds_key == "mnist" else 3
    img_size = 28 if ds_key == "mnist" else 32

    train_loader, val_loader, train_fid_loader = make_dataloader(ds_key, batch_size, img_size, channels, val_split=val_split)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TinyUNet(in_channels=channels, base=base, time_dim=time_dim, multi_task=(mode=="multi")).to(device)

    ddpm = DDPM(DiffusionConfig(timesteps=timesteps, beta_start=beta_start, beta_end=beta_end)).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    # Logging buffers
    train_steps = []
    tr_loss_main, tr_loss_x0, tr_loss_cons, tr_loss_total = [], [], [], []
    val_epochs, va_loss_main, va_loss_x0 = [], [], []
    fid_train_hist, fid_val_hist = [], []

    step = 0
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"[{mode}|{ds_key}] epoch {epoch+1}/{epochs}")
        for imgs, _ in pbar:
            imgs = imgs.to(device)
            t = torch.randint(0, ddpm.cfg.timesteps, (imgs.size(0),), device=device)
            total, parts = unified_loss(model, ddpm, imgs, t, multi_task=(mode=="multi"),
                                        w_x0=w_x0, w_consistency=w_consistency)
            optim.zero_grad(); total.backward(); optim.step()

            step += 1
            train_steps.append(step)
            tr_loss_main.append(float(parts["loss"]))
            if mode == "multi":
                tr_loss_x0.append(float(parts["loss_x0"]))
                tr_loss_cons.append(float(parts["loss_cons"]))
                tr_loss_total.append(float(parts["loss_total"]))
                pbar.set_postfix({"loss": f"{parts['loss']:.4f}", "tot": f"{parts['loss_total']:.4f}"})
            else:
                pbar.set_postfix({"loss": f"{parts['loss']:.4f}"})

            if step % sample_every == 0:
                with torch.no_grad():
                    grid_path = images_dir / f"samples_step{step}.png"
                    _path, _ = sample(
                        model, ddpm,
                        shape=(n_sample, channels, img_size, img_size),
                        device=device, save_path=str(grid_path),
                    )

        # --- End-of-epoch: Validation losses ---
        val_metrics = evaluate_mse_unified(model, ddpm, val_loader, device, multi_task=(mode=="multi"))
        va_loss_main.append(float(val_metrics["loss"]))
        if mode == "multi":
            va_loss_x0.append(float(val_metrics["loss_x0"]))

        # --- Prepare real images for FID (train & val) ---
        # Clean dirs
        for d in [fid_train_real, fid_train_fake, fid_val_real, fid_val_fake]:
            for f in d.glob("*.png"): f.unlink()

        # Real train images
        collected = 0; imgs_accum = []
        for imgs, _ in train_fid_loader:
            imgs_accum.append(imgs)
            collected += imgs.size(0)
            if collected >= fid_eval_images: break
        real_train = torch.cat(imgs_accum, dim=0)[:fid_eval_images]
        real_train = denorm(real_train, channels)
        dump_images(real_train, str(fid_train_real), prefix="real")

        # Real val images
        collected = 0; imgs_accum = []
        for imgs, _ in val_loader:
            imgs_accum.append(imgs)
            collected += imgs.size(0)
            if collected >= fid_eval_images: break
        real_val = torch.cat(imgs_accum, dim=0)[:fid_eval_images]
        real_val = denorm(real_val, channels)
        dump_images(real_val, str(fid_val_real), prefix="real")

        # --- Generate fake for train FID ---
        _, fake_batch = sample(
            model, ddpm,
            shape=(min(fid_eval_images, n_sample), channels, img_size, img_size),
            device=device, save_path=str(images_dir / f"samples_trainfid_epoch{epoch+1}.png"),
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
        dump_images(fake_train, str(fid_train_fake), prefix="fake")

        # --- Generate fake for val FID (fresh samples) ---
        _, fake_batch_val = sample(
            model, ddpm,
            shape=(min(fid_eval_images, n_sample), channels, img_size, img_size),
            device=device, save_path=str(images_dir / f"samples_valfid_epoch{epoch+1}.png"),
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
        dump_images(fake_val, str(fid_val_fake), prefix="fake")

        # --- Compute FIDs ---
        fid_train = compute_fid(str(fid_train_real), str(fid_train_fake), device)
        fid_val   = compute_fid(str(fid_val_real),   str(fid_val_fake),   device)
        fid_train_hist.append(float(fid_train))
        fid_val_hist.append(float(fid_val))

        # Epoch log to console
        if mode == "multi":
            print(f"[val] epoch {epoch+1}: loss={va_loss_main[-1]:.6f} | loss_x0={va_loss_x0[-1]:.6f} | "
                  f"FID_train={fid_train:.2f} | FID_val={fid_val:.2f}")
        else:
            print(f"[val] epoch {epoch+1}: loss={va_loss_main[-1]:.6f} | "
                  f"FID_train={fid_train:.2f} | FID_val={fid_val:.2f}")

        # Save curves/CSVs incrementally
        save_curves_unified(
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
        )

    # Final sample grid
    with torch.no_grad():
        path, _ = sample(
            model, ddpm,
            shape=(n_sample, channels, img_size, img_size),
            device=device, save_path=str(images_dir / "samples_final.png"),
        )
    print(f"Saved final samples to {path}")

    # Final save (redundant)
    save_curves_unified(
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
    )

# ---- Run the requested experiments ----
if __name__ == "__main__":
    # You can mount Drive in Colab if desired, then point save_root there.
    SAVE_ROOT = "/content/drive/MyDrive/prototypes/tiny_ddpm_mt/"

    # 1) Single-task MNIST
    train_unified(
        save_root=SAVE_ROOT, mode="single", data="mnist",
        epochs=10, timesteps=200, n_sample=64, sample_every=500,
    )

    # 2) Single-task CIFAR10
    train_unified(
        save_root=SAVE_ROOT, mode="single", data="cifar10",
        epochs=10, timesteps=200, n_sample=64, sample_every=500,
    )

    # 3) Multi-task MNIST and CIFAR10 (separate folders)
    for ds in ["mnist", "cifar10"]:
        train_unified(
            save_root=SAVE_ROOT, mode="multi", data=ds,
            epochs=10, timesteps=200, n_sample=64, sample_every=500,
            w_x0=1.0, w_consistency=0.1,
        )
