
import numpy as np
import pandas as pd


def encode_yes_no(X):
    
    X = X.copy()
    for col in X.columns:
        X[col] = X[col].map({'Yes': 1, 'No': 0})
    return X.astype(float)
