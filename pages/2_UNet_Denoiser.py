"""
pages/2_UNet_Denoiser.py
=========================
Streamlit page: U-Net-based denoiser. This file is auto-discovered by
Streamlit's multi-page app mechanism (any script inside pages/ next to
the main entry script becomes a page in the sidebar).

It simply reuses all the logic already defined in app_unet.py so there
is no duplicated code — we just call its main() function.
"""

import sys
import os

# Make sure the project root (where app_unet.py lives) is importable.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app_unet import main  # noqa: E402

main()