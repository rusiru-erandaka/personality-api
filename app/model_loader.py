"""
Model loader utilities for the serialized sklearn pipeline.
"""

from pathlib import Path
import sys

import joblib

from app.preprocessor import encode_yes_no

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "model.pkl"

pipeline = None
load_error = None

FEATURE_ORDER = [
    "Time_spent_Alone",
    "Stage_fear",
    "Social_event_attendance",
    "Going_outside",
    "Drained_after_socializing",
    "Friends_circle_size",
    "Post_frequency",
]


def _register_pickle_symbols() -> None:
    """
    Make the training-time preprocessing function visible to pickle.
    """
    for module_name in ("__main__", "__mp_main__", "run", "app.main"):
        module = sys.modules.get(module_name)
        if module is not None:
            setattr(module, "encode_yes_no", encode_yes_no)


def get_pipeline():
    """
    Load the model once and cache it for subsequent invocations.
    """
    global pipeline, load_error

    if pipeline is not None:
        return pipeline

    if not MODEL_PATH.exists():
        load_error = f"model.pkl not found at {MODEL_PATH}"
        raise FileNotFoundError(load_error)

    _register_pickle_symbols()

    try:
        pipeline = joblib.load(MODEL_PATH)
        load_error = None
        return pipeline
    except Exception as exc:
        load_error = f"{type(exc).__name__}: {exc}"
        print(f"Model load failed: {load_error}")
        raise
