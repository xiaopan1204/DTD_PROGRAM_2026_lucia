"""
tests/test_diffusion.py
========================
Part 4: Automated tests for the MNIST diffusion-denoising project.

These tests check the core building blocks (noise injection + the CNN
model) without needing a full training run or a downloaded dataset, so
they run fast and can be used as a quick regression check whenever the
code changes.

Run with:
    pytest
    # or, from the project root:
    pytest tests/test_diffusion.py -v
"""

import os
import sys

import numpy as np
import pytest
import torch

# Make sure we can import from the project root (train.py) no matter
# where pytest is invoked from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from train import SmallCNN, make_noisy  # noqa: E402


# -----------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------
@pytest.fixture
def small_image():
    """A tiny 8x8 'image' used to keep tests fast and independent of
    the real 28x28 MNIST resolution."""
    rng = np.random.default_rng(0)
    return rng.random((8, 8)).astype(np.float32)


@pytest.fixture
def mnist_sized_image():
    """A 28x28 image matching the real MNIST / model input resolution."""
    rng = np.random.default_rng(0)
    return rng.random((28, 28)).astype(np.float32)


@pytest.fixture
def cnn_model():
    return SmallCNN()


# -----------------------------------------------------------------
# 1. Image dimensions are correct
# -----------------------------------------------------------------
def test_mnist_sized_image_has_correct_dimensions(mnist_sized_image):
    assert mnist_sized_image.shape == (28, 28)
    assert mnist_sized_image.dtype == np.float32


def test_small_image_has_correct_dimensions(small_image):
    assert small_image.shape == (8, 8)


# -----------------------------------------------------------------
# 2. Adding noise returns an image of the same shape as the input
#    (explicitly checked for an 8x8 image, as requested)
# -----------------------------------------------------------------
def test_make_noisy_returns_8x8_image(small_image):
    noisy = make_noisy(small_image, alpha_t=0.3, seed=1)
    assert noisy.shape == (8, 8)
    assert noisy.dtype == np.float32


def test_make_noisy_preserves_shape_for_any_input(mnist_sized_image):
    noisy = make_noisy(mnist_sized_image, alpha_t=0.5, seed=2)
    assert noisy.shape == mnist_sized_image.shape


def test_make_noisy_changes_the_image(mnist_sized_image):
    # With non-trivial noise, the noisy image should not be identical
    # to the clean image.
    noisy = make_noisy(mnist_sized_image, alpha_t=0.3, seed=3)
    assert not np.allclose(noisy, mnist_sized_image)


def test_make_noisy_is_reproducible_with_same_seed(mnist_sized_image):
    noisy_a = make_noisy(mnist_sized_image, alpha_t=0.3, seed=42)
    noisy_b = make_noisy(mnist_sized_image, alpha_t=0.3, seed=42)
    np.testing.assert_array_equal(noisy_a, noisy_b)


# -----------------------------------------------------------------
# 3. The CNN produces the expected output shape
# -----------------------------------------------------------------
def test_cnn_output_shape_matches_input_shape(cnn_model, mnist_sized_image):
    x = torch.tensor(mnist_sized_image).unsqueeze(0).unsqueeze(0)  # (1,1,28,28)
    with torch.no_grad():
        output = cnn_model(x)
    assert output.shape == (1, 1, 28, 28)


def test_cnn_output_shape_matches_input_shape_for_a_batch(cnn_model):
    batch = torch.rand(4, 1, 28, 28)  # batch of 4 images
    with torch.no_grad():
        output = cnn_model(batch)
    assert output.shape == (4, 1, 28, 28)


# -----------------------------------------------------------------
# 4. Reconstruction returns valid numerical values (no NaN/Inf)
# -----------------------------------------------------------------
def test_cnn_reconstruction_has_valid_values(cnn_model, mnist_sized_image):
    noisy = make_noisy(mnist_sized_image, alpha_t=0.3, seed=5)
    z = torch.tensor(noisy).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        reconstructed = cnn_model(z).squeeze(0).squeeze(0).numpy()

    assert not np.isnan(reconstructed).any(), "Reconstruction contains NaN values"
    assert not np.isinf(reconstructed).any(), "Reconstruction contains infinite values"
    assert reconstructed.shape == mnist_sized_image.shape


def test_mse_between_reconstruction_and_clean_is_finite_and_nonnegative(cnn_model, mnist_sized_image):
    noisy = make_noisy(mnist_sized_image, alpha_t=0.3, seed=6)
    z = torch.tensor(noisy).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        reconstructed = cnn_model(z).squeeze(0).squeeze(0).numpy()

    mse = float(np.mean((reconstructed - mnist_sized_image) ** 2))
    assert np.isfinite(mse)
    assert mse >= 0.0
