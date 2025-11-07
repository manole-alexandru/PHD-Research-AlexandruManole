from __future__ import annotations
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import os


def _default_loader_kwargs():
    nw = os.cpu_count() or 2
    nw = max(0, min(4, nw // 2))
    pin = torch.cuda.is_available()
    # persistent_workers speeds up on repeated epochs when nw > 0
    return dict(num_workers=nw, pin_memory=pin, persistent_workers=(nw > 0))


def make_dataloader(name: str, batch_size: int, img_size: int, channels: int, val_split: float = 0.05):
    name_l = name.lower()
    if name_l == "mnist":
        tfm = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])
        ds_base = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    elif name_l in ["cifar10", "cifar"]:
        tfm = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3),
        ])
        ds_base = datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm)
    elif name_l == "cifar100":
        tfm = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3),
        ])
        ds_base = datasets.CIFAR100(root="./data", train=True, download=True, transform=tfm)
    elif name_l == "svhn":
        tfm = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3),
        ])
        ds_base = datasets.SVHN(root="./data", split="train", download=True, transform=tfm)
    elif name_l == "celeba":
        tfm = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3),
        ])
        ds_base = datasets.CelebA(root="./data", split="train", download=True, transform=tfm)
    else:
        raise ValueError("Unsupported dataset. Use one of: mnist, cifar10, cifar100, svhn, celeba.")

    val_size = max(1, int(len(ds_base) * val_split))
    train_size = len(ds_base) - val_size
    train_ds, val_ds = torch.utils.data.random_split(ds_base, [train_size, val_size])

    kwargs = _default_loader_kwargs()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kwargs)
    train_fid_loader = DataLoader(ds_base, batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, val_loader, train_fid_loader


def make_test_loader(name: str, batch_size: int, img_size: int, channels: int):
    name_l = name.lower()
    if name_l == "mnist":
        tfm = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])
        ds_test = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    elif name_l in ["cifar10", "cifar"]:
        tfm = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3),
        ])
        ds_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm)
    elif name_l == "cifar100":
        tfm = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3),
        ])
        ds_test = datasets.CIFAR100(root="./data", train=False, download=True, transform=tfm)
    elif name_l == "svhn":
        tfm = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3),
        ])
        ds_test = datasets.SVHN(root="./data", split="test", download=True, transform=tfm)
    elif name_l == "celeba":
        tfm = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3),
        ])
        ds_test = datasets.CelebA(root="./data", split="test", download=True, transform=tfm)
    else:
        raise ValueError("Unsupported dataset. Use one of: mnist, cifar10, cifar100, svhn, celeba.")
    kwargs = _default_loader_kwargs()
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, **kwargs)
    return test_loader
