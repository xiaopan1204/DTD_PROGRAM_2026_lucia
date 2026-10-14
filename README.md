# MNIST Digit Denoising (Linear Regression → CNN → U-Net)

A small diffusion-style denoising project built around the forward noising process

```
z_t = sqrt(alpha_t) * x + sqrt(1 - alpha_t) * eps,   eps ~ N(0, I)
```

The project takes a proof-of-concept model (linear regression baseline → CNN) and
develops it into a reproducible, tested, containerized, and deployed application,
following the full ML project lifecycle: **Build → Improve → Organise → Reproduce →
Test → Package → Automate → Deploy**.

An additional **U-Net** model is included as a further improvement over the baseline CNN.

---

## Project Structure

```
DTD_PROGRAM_2026_lucia/
├── train.py                  # trains the CNN denoiser -> saves model.pt
├── train_unet.py              # trains the U-Net denoiser -> saves model_unet.pt
├── model.pt                   # trained CNN weights
├── model_unet.pt               # trained U-Net weights
├── Home.py                    # Streamlit entry point (multi-page app)
├── pages/
│   ├── 1_CNN_Denoiser.py      # Streamlit page using the CNN model
│   └── 2_UNet_Denoiser.py      # Streamlit page using the U-Net model
├── app.py                     # CNN inference + Streamlit UI logic
├── app_unet.py                 # U-Net inference + Streamlit UI logic
├── tests/
│   └── test_diffusion.py       # pytest unit tests
├── .github/workflows/
│   └── test.yml                # GitHub Actions CI (runs pytest on every push)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
├── .dockerignore
└── README.md
```

---

## 1. Install and Run Locally

```bash
git clone https://github.com/xiaopan1204/DTD_PROGRAM_2026_lucia.git
cd DTD_PROGRAM_2026_lucia
python3 -m pip install -r requirements.txt
```

Train the models (downloads MNIST automatically into `mnist_data/` if not present):

```bash
python3 train.py --epochs 15          # produces model.pt (CNN)
python3 train_unet.py --epochs 15     # produces model_unet.pt (U-Net)
```

Run the app (multi-page: sidebar lets you switch between CNN and U-Net):

```bash
python3 -m streamlit run Home.py
```

Open the URL Streamlit prints (default `http://localhost:8501`) in your browser.

---

## 2. Run the Tests

```bash
pip install pytest
pytest tests/ -v
```

The tests check:
- image dimensions are correct;
- adding noise returns an image of the expected shape (including an 8x8 case);
- the CNN produces the expected output shape;
- reconstruction returns valid numerical values (no NaN/Inf).

---

## 3. Build and Run with Docker

This project is built and run with **`docker compose`**, which uses the
`Dockerfile` under the hood and also orchestrates training. Four services
are defined: `train`, `train-unet` (one-off training jobs) and `app`
(the Streamlit demo).

**Important — run the services in this order** the first time, since the
app needs a real `model.pt` / `model_unet.pt` file to already exist on
the host before it can be mounted into the container:

```bash
# 1) Build the image and train the CNN (writes model.pt to the host)
docker compose run --rm train

# 2) Train the U-Net (writes model_unet.pt to the host)
docker compose run --rm train-unet

# 3) Run the app (uses the model files produced above)
docker compose up --build app
```

Open `http://localhost:8501` in your browser.

> If you add or change files (e.g. `Home.py`, `pages/`) and `docker compose up`
> still shows the old version, force a rebuild with `docker compose up --build`
> — otherwise Compose may reuse a stale cached image.

---

## 4. Continuous Integration (GitHub Actions)

Every push triggers `.github/workflows/test.yml`, which installs dependencies and
runs the pytest suite automatically. See the **Actions** tab of this repository for
the latest run status (pass/fail).

---

## 5. Live Deployment

The app is deployed on Streamlit Community Cloud and can be accessed here:

```
https://mnist-denoiser-lucia.streamlit.app
```

---

## Model Summary

| Model | Architecture | Test MSE (alpha_t = 0.3) |
|---|---|---|
| Linear Regression (baseline) | per-pixel OLS | ~0.061 |
| CNN | 4 conv layers (stacked 3x3) | ~0.054 |
| U-Net | encoder/decoder + skip connections | ~0.042 |

See the project report for full discussion and reconstructed-image comparisons.
