"""
train_unet.py
=============
Trains a small U-Net denoiser on noisy MNIST observations
        z_t = sqrt(alpha_t) * x + sqrt(1 - alpha_t) * eps,   eps ~ N(0, I)
and saves the trained weights to model_unet.pt.

U-Net vs. the plain SmallCNN (train.py):
  - Encoder downsamples (28->14->7) to capture larger-scale / global
    structure (e.g. the overall stroke layout of a digit).
  - Decoder upsamples back to 28x28.
  - Skip connections pass fine-grained spatial detail straight from the
    encoder to the decoder, so the network keeps both global context and
    local detail. This is the same architecture family used as the
    denoising backbone in real diffusion models (e.g. DDPM).

Workflow:  train_unet.py -> model_unet.pt -> app.py (load this checkpoint)

Usage:
    python train_unet.py
    python train_unet.py --epochs 20 --alpha 0.3 --n_train 10000

If the MNIST dataset is not found locally (in ./mnist_data), it will be
downloaded automatically via torchvision.
"""

import os
import shutil
import tempfile
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
MODEL_PATH = os.path.join(BASE_DIR, "model_unet.pt")


# -----------------------------------------------------------------
# Model definition: small U-Net
#   (must match the architecture used wherever model_unet.pt is loaded)
# -----------------------------------------------------------------
class UNetSmall(nn.Module):
    """Small U-Net denoiser: encoder (down-sampling) -> bottleneck ->
    decoder (up-sampling) with skip connections. Operates on 1x28x28
    MNIST images and predicts the clean image x from a noisy z_t."""

    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
                                   nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2)  # 28 -> 14

        self.enc2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                                   nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2)  # 14 -> 7

        # Bottleneck
        self.bottleneck = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                                         nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())

        # Decoder
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)  # 7 -> 14
        self.dec2 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),  # 64 = 32(up)+32(skip)
                                   nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())

        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)  # 14 -> 28
        self.dec1 = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),  # 32 = 16(up)+16(skip)
                                   nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())

        self.out_conv = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)          # (B,16,28,28)
        p1 = self.pool1(e1)        # (B,16,14,14)

        e2 = self.enc2(p1)         # (B,32,14,14)
        p2 = self.pool2(e2)        # (B,32,7,7)

        b = self.bottleneck(p2)    # (B,64,7,7)

        u2 = self.up2(b)           # (B,32,14,14)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))  # skip connection

        u1 = self.up1(d2)          # (B,16,28,28)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))  # skip connection

        return self.out_conv(d1)   # (B,1,28,28)


# -----------------------------------------------------------------
# Data utilities
# -----------------------------------------------------------------
def load_mnist(n_train, n_test):
    """Load MNIST from DATA_DIR; download automatically if not present."""
    already_downloaded = os.path.exists(os.path.join(DATA_DIR, "MNIST", "raw"))
    if already_downloaded:
        print(f"[train_unet.py] Found local MNIST data in '{DATA_DIR}', skipping download.")
    else:
        print(f"[train_unet.py] MNIST data not found locally, downloading to '{DATA_DIR}' ...")

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
    print(f"[train_unet.py] Using device: {device}")

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

    model = UNetSmall().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_test_mse = float("inf")
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
        best_test_mse = min(best_test_mse, test_mse)

        print(f"Epoch {epoch+1:2d}/{args.epochs}  train MSE={train_mse:.5f}  test MSE={test_mse:.5f}")

    # Save model weights + the alpha_t used, so app.py can reproduce
    # exactly the same noise process at inference time.
    # NOTE: .pt files are zip archives that require random-access (seek)
    # writes, which some network-mounted filesystems don't support well.
    # We save to a local temp file first, then copy it into place.
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "alpha_t": args.alpha,
        "architecture": "UNetSmall",
        "final_test_mse": test_mse,
        "best_test_mse": best_test_mse,
    }
    tmp_path = os.path.join(tempfile.gettempdir(), "model_unet_tmp.pt")
    torch.save(checkpoint, tmp_path)
    shutil.copy(tmp_path, MODEL_PATH)
    print(f"[train_unet.py] Model saved to {MODEL_PATH}")
    print(f"[train_unet.py] Final test MSE = {test_mse:.5f} (best during training: {best_test_mse:.5f})")


def parse_args():
    p = argparse.ArgumentParser(description="Train a small U-Net MNIST denoiser.")
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
