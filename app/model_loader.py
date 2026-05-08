"""
Model loader — loads the sklearn pipeline once at startup
and exposes it as a module-level singleton.
"""

import joblib
from pathlib import Path

# ── IMPORTANT: import before joblib.load() ───────────────────────────────
# pickle needs to resolve encode_yes_no which was embedded in the
# sklearn FunctionTransformer during training. Importing it here
# registers it in the module namespace so pickle can find it.
from app.preprocessor import encode_yes_no  # noqa: F401

# ── Resolve model path ────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "model.pkl"

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
