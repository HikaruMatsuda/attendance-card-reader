"""MNIST に強いアフィン拡張（傾き・回転・拡縮・平行移動・太さ）をかけて小型 CNN を学習し、ONNX に書き出す。

手書きの学籍番号は強く傾いた字や細い字が多く、既製の MNIST モデルでは誤読が多いため。
使い方: python tools/train_digits.py  （torch が必要。推論側は onnxruntime のみ）
"""

from __future__ import annotations

import gzip
import math
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
         "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
DATA = Path(__file__).parent / "_mnist"
OUT = Path(__file__).parent.parent / "models" / "digits_aug.onnx"


def load() -> tuple[torch.Tensor, ...]:
    DATA.mkdir(exist_ok=True)
    arrs = []
    for f in FILES:
        p = DATA / f
        if not p.exists():
            urllib.request.urlretrieve(BASE + f, p)
        raw = gzip.open(p).read()
        a = np.frombuffer(raw, np.uint8, offset=16 if "images" in f else 8)
        arrs.append(a.reshape(-1, 28, 28) if "images" in f else a)
    xtr, ytr, xte, yte = arrs
    t = lambda x: torch.from_numpy(x.copy())
    return t(xtr).float() / 255, t(ytr).long(), t(xte).float() / 255, t(yte).long()


def augment(x: torch.Tensor) -> torch.Tensor:
    """x: (B,1,28,28)。傾き±0.6 / 回転±20° / 拡縮 / 移動 / 太さ変化。"""
    b = x.shape[0]
    r = lambda lo, hi: torch.empty(b).uniform_(lo, hi)
    ang, shear = r(-20, 20) * math.pi / 180, r(-0.6, 0.6)
    sx, sy = r(0.8, 1.15), r(0.8, 1.15)
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.zeros(b, 2, 3)
    theta[:, 0, 0] = cos / sx
    theta[:, 0, 1] = (-sin + shear) / sx
    theta[:, 1, 0] = sin / sy
    theta[:, 1, 1] = cos / sy
    theta[:, :, 2] = torch.empty(b, 2).uniform_(-0.12, 0.12)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
    # 太さ: 一部を膨張、一部を細く（ガンマ・しきい値で近似）
    mode = torch.randint(0, 3, (b, 1, 1, 1))
    thick = F.max_pool2d(x, 3, 1, 1)
    thin = torch.clamp((x - 0.35) / 0.65, 0, 1) ** 1.5
    x = torch.where(mode == 0, thick, torch.where(mode == 1, thin, x))
    return x.clamp(0, 1)


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(128 * 9, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x)


def main() -> None:
    torch.manual_seed(0)
    xtr, ytr, xte, yte = load()
    xtr, xte = xtr.unsqueeze(1), xte.unsqueeze(1)
    net = Net()
    epochs = 12
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    steps = epochs * (len(xtr) // 256)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-3, total_steps=steps)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr) - 255, 256):
            idx = perm[i : i + 256]
            loss = F.cross_entropy(net(augment(xtr[idx])), ytr[idx], label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        net.eval()
        with torch.no_grad():
            acc = (net(xte).argmax(1) == yte).float().mean().item()
            acc_aug = (net(augment(xte)).argmax(1) == yte).float().mean().item()
        print(f"epoch {ep + 1}: test acc {acc:.4f} / augmented test acc {acc_aug:.4f}", flush=True)
    OUT.parent.mkdir(exist_ok=True)
    torch.save(net.state_dict(), DATA / "net.pt")
    torch.onnx.export(net, torch.zeros(1, 1, 28, 28), OUT, input_names=["input"],
                      output_names=["logits"], dynamic_axes={"input": {0: "n"}}, dynamo=False)
    print("saved", OUT)


if __name__ == "__main__":
    main()
