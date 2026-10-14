"""
Home.py
=======
Entry point for the multi-page Streamlit app. Streamlit automatically
turns any .py files inside the sibling `pages/` folder into navigable
pages, shown in the sidebar — no manual routing/proxy setup needed.

Run with:
    streamlit run Home.py
"""

import streamlit as st

st.set_page_config(page_title="MNIST Denoiser", page_icon="🔢", layout="centered")

st.title("MNIST Digit Denoiser")
st.write(
    """
    This app demonstrates denoising a noisy MNIST-style digit image using
    two different model architectures:

    - **CNN Denoiser** — a small stacked-convolution network
    - **U-Net Denoiser** — an encoder/decoder network with skip connections

    👉 Use the sidebar on the left to switch between the two pages.
    """
)

st.info("Select a page from the sidebar (◀ top-left) to get started.")