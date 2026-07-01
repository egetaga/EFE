"""
Tabular Feature Evolution v2 — Evolved Feature Program (fit/transform)

Rules:
  - fit(X, y) receives training features + target, returns a state dict
  - transform(X, state) uses fitted state to generate new features
  - Keep ALL original columns unchanged
  - Append new generated columns alongside originals
  - Preserve row count, row order, index, and NaN values
  - Do NOT mutate input DataFrames
  - State dict must contain only simple Python types (int, float, str,
    list, dict, numpy arrays)
  - No imputation, no scaling originals, no PCA
  - No training ML models inside fit()
  - Max 50 generated features
"""

import numpy as np
import pandas as pd


class FeatureProgram:
    def __init__(self, seed=42):
        self.seed = seed

    # EVOLVE-BLOCK-START
    def fit(self, X, y):
        """Fit feature parameters from training data.

        Parameters
        ----------
        X : pd.DataFrame
            Training features (may contain NaN).
        y : pd.Series
            Training target (binary: 0/1).

        Returns
        -------
        state : dict
            Fitted parameters to pass to transform().
        """
        state = {}
        return state

    def transform(self, X, state):
        """Generate new features using fitted state.

        Parameters
        ----------
        X : pd.DataFrame
            Input features (train, val, or test).
        state : dict from fit()

        Returns
        -------
        X_out : pd.DataFrame
            Original columns + generated columns.
        """
        return X.copy()
    # EVOLVE-BLOCK-END


def build_feature_program(seed=42):
    return FeatureProgram(seed=seed)
