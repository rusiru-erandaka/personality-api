"""
Shared preprocessing functions.

This module must be imported BEFORE joblib.load() is called,
so that pickle can resolve the custom transformer function
that was used during model training.
"""

import numpy as np
import pandas as pd


def encode_yes_no(X):
    """
    Encodes Yes/No string columns to 1/0 floats.
    NaN values are preserved for the downstream imputer to handle.
    """
    X = X.copy()
    for col in X.columns:
        X[col] = X[col].map({'Yes': 1, 'No': 0})
    return X.astype(float)
