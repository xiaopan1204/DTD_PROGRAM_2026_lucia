"""
train.py
========
Trains a small CNN denoiser on noisy MNIST observations
        z_t = sqrt(alpha_t) * x + sqrt(1 - alpha_t) * eps,   eps ~ N(0, I)
and saves the trained weights to model.pt.

Workflow:  train.py -> model.pt -> app.py

Usage:
    python train.py
    python train.py --epochs 20 --alpha 0.3 --n_train 10000

If the MNIST dataset is not found locally (in ./mnist_data), it will be
downloaded automatically via torchvision.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision

# -----------------------------------------------------------------
# Paths
# -----------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "mnist_data")
MODEL_PATH = os.path.join(BASE_DIR, "model.pt")


# -----------------------------------------------------------------
# Model definition (must match the architecture used in app.py)
# -----------------------------------------------------------------
class SmallCNN(nn.Module):
    """Small CNN denoiser: predicts the clean image x from a noisy
    observation z_t. Operates directly on 1x28x28 MNIST images."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------------------------------------------
# Data utilities
# -----------------------------------------------------------------
def load_mnist(n_train, n_test):
    """Load MNIST from DATA_DIR; download automatically if not present."""
    already_downloaded = os.path.exists(os.path.join(DATA_DIR, "MNIST", "raw"))
    if already_downloaded:
        print(f"[train.py] Found local MNIST data in '{DATA_DIR}', skipping download.")
    else:
        print(f"[train.py] MNIST data not found locally, downloading to '{DATA_DIR}' ...")

    train_ds = torchvision.datasets.MNIST(root=DATA_DIR, train=True, download=True)
    test_ds = torchvision.datasets.MNIST(root=DATA_DIR, train=False, download=True)

    X_train = train_ds.data[:n_train].float().numpy() / 255.0
    X_test = test_ds.data[:n_test].float().numpy() / 255.0
    return X_train, X_test


def make_noisy(x, alpha_t, seed=None):
    """z_t = sqrt(alpha_t) * x + sqrt(1 - alpha_t) * eps"""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(x.shape).astype(np.float32)
    z = np.sqrt(alpha_t) * x + np.sqrt(1 - alpha_t) * eps
    return z.astype(np.float32)


# -----------------------------------------------------------------
# Training
# -----------------------------------------------------------------
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train.py] Using device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X_train, X_test = load_mnist(args.n_train, args.n_test)
    Z_train = make_noisy(X_train, args.alpha, seed=1)
    Z_test = make_noisy(X_test, args.alpha, seed=2)

    Ztr_t = torch.tensor(Z_train).unsqueeze(1)
    Xtr_t = torch.tensor(X_train).unsqueeze(1)
    Zte_t = torch.tensor(Z_test).unsqueeze(1).to(device)
    Xte_t = torch.tensor(X_test).unsqueeze(1).to(device)

    loader = DataLoader(TensorDataset(Ztr_t, Xtr_t), batch_size=args.batch_size, shuffle=True)

    model = SmallCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for zb, xb in loader:
            zb, xb = zb.to(device), xb.to(device)
            opt.zero_grad()
            pred = model(zb)
            loss = F.mse_loss(pred, xb)
            loss.backward()
            opt.step()
            running += loss.item() * zb.size(0)
        train_mse = running / len(X_train)

        model.eval()
        with torch.no_grad():
            test_mse = F.mse_loss(model(Zte_t), Xte_t).item()

        print(f"Epoch {epoch+1:2d}/{args.epochs}  train MSE={train_mse:.5f}  test MSE={test_mse:.5f}")

    # Save model weights + the alpha_t used, so app.py can reproduce
    # exactly the same noise process at inference time.
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "alpha_t": args.alpha,
            "architecture": "SmallCNN",
        },
        MODEL_PATH,
    )
    print(f"[train.py] Model saved to {MODEL_PATH}")


def parse_args():
    p = argparse.ArgumentParser(description="Train a small CNN MNIST denoiser.")
    p.add_argument("--epochs", type=int, default=15, help="number of training epochs")
    p.add_argument("--alpha", type=float, default=0.3, help="diffusion coefficient alpha_t")
    p.add_argument("--n_train", type=int, default=8000, help="number of training images")
    p.add_argument("--n_test", type=int, default=1000, help="number of test images")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
