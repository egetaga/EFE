"""
Time-Series Transform Evolution — Evolved Transform Program

Rules:
  - fit() receives a 1-D numpy array of historical target values and a meta dict
  - fit() returns a state dict of fitted parameters (must be serialisable)
  - fit() MUST store "hist_len": len(y_hist) in the state dict
  - transform() applies the forward transform using fitted state
  - inverse_transform() exactly reverses transform (round-trip error < 1e-4)
  - inverse_transform() is called on arrays of DIFFERENT lengths than transform():
    transform() receives the full history (length T), but inverse_transform()
    receives the forecast (length prediction_length). For position-dependent
    transforms (e.g., detrending), use state["hist_len"] to compute the correct
    time offset, NOT len(z).
  - Do NOT use future data — fit only from the history passed in
  - Do NOT mutate input arrays — copy first
  - Preserve length, order, and NaN positions
  - Output must be finite (no inf); NaN only where input had NaN
  - inverse_transform must remain stable slightly beyond the observed range
  - Keep computation lightweight — called once per series per rolling window
  - Be deterministic given the same input
"""

import numpy as np


class TransformProgram:
    def __init__(self, seed=42):
        self.seed = seed

    # EVOLVE-BLOCK-START
    def fit(self, y_hist, meta=None):
        """Fit transform parameters from historical target values.

        Parameters
        ----------
        y_hist : np.ndarray, shape (T,)
            Historical target values (may contain NaN).
        meta : dict or None
            Dataset-level metadata (see evaluator for keys).

        Returns
        -------
        state : dict
            Fitted parameters to pass to transform / inverse_transform.
        """
        state = {"family": "identity", "hist_len": len(y_hist)}
        return state

    def transform(self, y, state):
        """Apply forward transform.

        Parameters
        ----------
        y : np.ndarray, shape (N,)
        state : dict from fit()

        Returns
        -------
        z : np.ndarray, shape (N,)
        """
        return y.copy()

    def inverse_transform(self, z, state):
        """Apply inverse transform (must exactly reverse transform).

        Parameters
        ----------
        z : np.ndarray, shape (N,)
        state : dict from fit()

        Returns
        -------
        y : np.ndarray, shape (N,)
        """
        return z.copy()
    # EVOLVE-BLOCK-END


def build_transform_program(seed=42):
    return TransformProgram(seed=seed)


def get_transform_program(seed=42):
    return build_transform_program(seed=seed)
