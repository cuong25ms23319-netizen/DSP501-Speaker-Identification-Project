"""
Module 3: preemphasis.py
------------------------
Apply pre-emphasis filtering to speech signals.

This is typically used before feature extraction (e.g., MFCC) to:
  - Emphasize high-frequency components
  - Flatten the speech spectrum
  - Improve discriminability for speaker / phoneme recognition
"""

import numpy as np


def pre_emphasize(y, alpha: float = 0.97):
    """
    Apply a first-order pre-emphasis filter to a 1-D signal.

    y'[n] = y[n] - alpha * y[n-1]

    Parameters
    ----------
    y : 1-D numpy array
        Input audio signal.
    alpha : float, optional
        Pre-emphasis coefficient in [0, 1). Typical values: 0.95–0.97.

    Returns
    -------
    y_emph : 1-D numpy array
        Pre-emphasized signal, same length as input.
    """
    if y.size == 0:
        return y

    # Use vectorized implementation for efficiency
    return np.concatenate(([y[0]], y[1:] - alpha * y[:-1]))


__all__ = ["pre_emphasize"]

