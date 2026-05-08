"""
Model loader — loads the sklearn pipeline once at startup
and exposes it as a module-level singleton.
"""

import os
import joblib
from pathlib import Path

# ── Resolve model path ────────────────────────────────────────────────────
# Looks for model.pkl in the /model directory at project root.
BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "model.pkl"

# ── Load on import (Vercel spins up a fresh process per cold start) ────────
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"model.pkl not found at {MODEL_PATH}. "
        "Please place your trained model file in the /model directory."
    )

pipeline = joblib.load(MODEL_PATH)

FEATURE_ORDER = [
    "Time_spent_Alone",
    "Stage_fear",
    "Social_event_attendance",
    "Going_outside",
    "Drained_after_socializing",
    "Friends_circle_size",
    "Post_frequency",
]
