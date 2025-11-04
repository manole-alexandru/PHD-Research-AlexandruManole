from __future__ import annotations
import math
import torch
import torch.nn as nn


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
            return self.out_eps(u2)

