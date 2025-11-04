from __future__ import annotations
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch


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

    val_size = max(1, int(len(ds_train) * val_split))
    train_size = len(ds_train) - val_size
    train_ds, val_ds = torch.utils.data.random_split(ds_train, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    train_fid_loader = DataLoader(ds_full_for_train_fid, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, train_fid_loader


def make_test_loader(name: str, batch_size: int, img_size: int, channels: int):
    tfm = [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
    if channels == 3:
        tfm = [transforms.Resize(img_size), transforms.ToTensor(),
               transforms.Normalize((0.5,)*3, (0.5,)*3)]
    tfm = transforms.Compose(tfm)
    if name.lower() == "mnist":
        ds_test = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    elif name.lower() in ["cifar10", "cifar"]:
        name = "cifar10"
        ds_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm)
    else:
        raise ValueError("Unsupported dataset. Use 'mnist' or 'cifar10'.")
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return test_loader
