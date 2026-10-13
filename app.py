"""
app.py
======
Streamlit interface for the MNIST denoiser.

Workflow:
  1. Load the trained SmallCNN denoiser from model.pt (produced by train.py).
  2. Let the user upload an image of a digit "6" (a cropped MNIST-like
     digit image works best; any image is auto-converted to grayscale
     and resized to 28x28).
  3. Add forward-diffusion noise:
         z_t = sqrt(alpha_t) * x + sqrt(1 - alpha_t) * eps,  eps ~ N(0, I)
  4. Run the trained CNN on z_t to reconstruct x_hat.
  5. Display: original (uploaded) / noisy / reconstructed images,
     plus the MSE between the reconstruction and the uploaded image.

Run with:
    streamlit run app.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import streamlit as st

# -----------------------------------------------------------------
# Paths
# -----------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pt")


# -----------------------------------------------------------------
# Model definition — MUST match the architecture used in train.py
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


@st.cache_resource
def load_model():
    """Load the trained model + the alpha_t it was trained with."""
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Could not find a trained model at '{MODEL_PATH}'. "
            f"Please run `python train.py` first to produce model.pt."
        )
        st.stop()

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    model = SmallCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    alpha_t = checkpoint.get("alpha_t", 0.3)
    return model, alpha_t


def preprocess_uploaded_image(uploaded_file, debug=False):
    """Convert an arbitrary uploaded photo/scan into a 28x28 MNIST-style
    array in [0,1] (centered digit, white stroke on black background).

    A naive resize-the-whole-photo-to-28x28 fails badly for real photos:
    the digit only occupies a small region, so most pixels are background
    and the digit gets blurred away. This function instead:
      1. Converts to grayscale.
      2. Auto-inverts if needed so the digit is brighter than the background.
      3. Binarizes with an adaptive threshold (Otsu-like) to separate the
         digit from background/shadows/lighting gradients.
      4. Crops to the digit's bounding box (with a small margin).
      5. Pads to a square and resizes to 20x20, then pastes onto a 28x28
         canvas with the digit centered by its center of mass — this
         mirrors the official MNIST preprocessing pipeline.
    """
    img = Image.open(uploaded_file).convert("L")  # grayscale
    arr = np.array(img).astype(np.float32) / 255.0

    # Normalize contrast: stretch to use the full [0,1] range first.
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    if hi > lo:
        arr = np.clip((arr - lo) / (hi - lo), 0, 1)

    # If the image is mostly bright (light background), assume the digit
    # is the darker ink and invert so the digit becomes bright (MNIST style).
    if arr.mean() > 0.5:
        arr = 1.0 - arr

    # Simple global threshold (Otsu) to binarize digit vs. background.
    hist, bin_edges = np.histogram((arr * 255).astype(np.uint8), bins=256, range=(0, 255))
    hist = hist.astype(np.float64)
    total = hist.sum()
    sum_all = np.dot(np.arange(256), hist)
    sum_bg, w_bg, best_thresh, best_var = 0.0, 0.0, 0, 0.0
    for t in range(256):
        w_bg += hist[t]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / w_bg
        mean_fg = (sum_all - sum_bg) / w_fg
        var_between = w_bg * w_fg * (mean_bg - mean_fg) ** 2
        if var_between > best_var:
            best_var = var_between
            best_thresh = t
    thresh = best_thresh / 255.0
    mask = arr > thresh

    # Find the digit's bounding box; fall back to the full image if the
    # threshold found nothing (e.g. a blank/near-uniform image).
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        cropped = arr
    else:
        pad = 2
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, arr.shape[0])
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, arr.shape[1])
        cropped = arr[y0:y1, x0:x1]

    # Pad the crop to a square (so resizing doesn't distort the digit's
    # aspect ratio), then resize to 20x20 (MNIST digits occupy ~20x20
    # inside the 28x28 canvas) and paste centered on a 28x28 black canvas.
    h, w = cropped.shape
    side = max(h, w)
    square = np.zeros((side, side), dtype=np.float32)
    y_off, x_off = (side - h) // 2, (side - w) // 2
    square[y_off:y_off + h, x_off:x_off + w] = cropped

    square_img = Image.fromarray((square * 255).astype(np.uint8))
    resized = square_img.resize((20, 20), Image.LANCZOS)
    resized_arr = np.array(resized).astype(np.float32) / 255.0

    canvas = np.zeros((28, 28), dtype=np.float32)
    canvas[4:24, 4:24] = resized_arr

    if debug:
        st.caption(f"Otsu threshold used: {thresh:.2f}")

    return canvas


def add_noise(x, alpha_t, seed=None):
    """Forward-diffusion noising: z_t = sqrt(alpha_t) x + sqrt(1-alpha_t) eps"""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(x.shape).astype(np.float32)
    z = np.sqrt(alpha_t) * x + np.sqrt(1 - alpha_t) * eps
    return z.astype(np.float32)


def denoise(model, z):
    """Run the trained CNN on a noisy 28x28 image and return the
    reconstructed 28x28 image as a numpy array."""
    with torch.no_grad():
        z_t = torch.tensor(z, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,28,28)
        x_hat = model(z_t).squeeze(0).squeeze(0).numpy()
    return x_hat


def mse(a, b):
    return float(np.mean((a - b) ** 2))


# -----------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------
def main():
    st.set_page_config(page_title="MNIST Digit Denoiser", layout="centered")
    st.title("MNIST Digit Denoiser — Digit \"6\" Demo")
    st.write(
        "Upload an image of a handwritten digit **6**, add diffusion noise, "
        "then reconstruct it with a trained CNN denoiser and compare the "
        "result to the original using MSE."
    )

    model, trained_alpha_t = load_model()

    with st.sidebar:
        st.header("Settings")
        alpha_t = st.slider(
            "alpha_t (noise level, higher = less noise)",
            min_value=0.05, max_value=0.95, value=float(trained_alpha_t), step=0.05,
        )
        seed = st.number_input("Random seed for noise", min_value=0, value=42, step=1)
        show_debug = st.checkbox("Show preprocessing debug info", value=False)
        st.caption(f"Model was trained with alpha_t = {trained_alpha_t:.2f}.")

    uploaded_file = st.file_uploader(
        "Upload an image of digit 6 (png/jpg)", type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is None:
        st.info("Please upload an image of a handwritten digit 6 to get started.")
        return

    # 1) Preprocess uploaded image -> clean 28x28 array
    #    (crops to the digit's bounding box, binarizes, and centers it
    #    like the official MNIST preprocessing, instead of naively
    #    resizing the whole photo)
    x_clean = preprocess_uploaded_image(uploaded_file, debug=show_debug)

    # 2) Add noise
    z_noisy = add_noise(x_clean, alpha_t, seed=int(seed))

    # 3) Reconstruct with the trained CNN
    x_hat = denoise(model, z_noisy)

    # 4) Compute MSE (reconstruction vs. the uploaded "clean" image)
    recon_mse = mse(x_hat, x_clean)
    noisy_mse = mse(z_noisy, x_clean)

    # 5) Display results
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Uploaded (clean)")
        st.image(x_clean, clamp=True, width=180)
    with col2:
        st.subheader("Noisy (z_t)")
        st.image(np.clip(z_noisy, 0, 1), clamp=True, width=180)
    with col3:
        st.subheader("Reconstructed")
        st.image(np.clip(x_hat, 0, 1), clamp=True, width=180)

    st.markdown("---")
    st.subheader("Results")
    c1, c2 = st.columns(2)
    c1.metric("MSE: noisy vs. clean", f"{noisy_mse:.5f}")
    c2.metric("MSE: reconstructed vs. clean", f"{recon_mse:.5f}")

    improvement = (noisy_mse - recon_mse) / noisy_mse * 100 if noisy_mse > 0 else 0.0
    if recon_mse < noisy_mse:
        st.success(f"The CNN denoiser reduced the MSE by {improvement:.1f}% compared to the noisy image.")
    else:
        st.warning(
            "The reconstruction did not improve on the noisy image for this input "
            "(this can happen with inputs that look very different from MNIST digits, "
            "or with a very low noise level)."
        )


if __name__ == "__main__":
    main()