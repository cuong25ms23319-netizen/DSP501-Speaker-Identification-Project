"""
Module 1: preprocess.py
-----------------------
Load and standardize audio files for the speaker identification pipeline.

Steps per file:
  1. Load mono audio at 16 kHz
  2. Normalize amplitude to [-1, 1]
  3. Trim leading/trailing silence
  4. Pad or crop to a fixed 3-second length (48000 samples)
"""

import numpy as np
import librosa


def load_audio(path, sr=16000):
    """Load a .wav file as mono at the given sample rate."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y, sr


def normalize(y):
    """Scale signal so the peak absolute value is 1.0."""
    peak = np.max(np.abs(y))
    if peak == 0:
        return y  # silence — return as-is to avoid division by zero
    return y / peak


def trim_silence(y, top_db=20):
    """Remove leading and trailing silence below top_db threshold."""
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return y_trimmed


def pad_or_crop(y, target_len=48000):
    """
    Make signal exactly target_len samples long.
    - If shorter: zero-pad on the right.
    - If longer:  crop from the start.
    """
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y


def preprocess(path, sr=16000, target_len=48000):
    """Full preprocessing pipeline for one audio file."""
    y, sr = load_audio(path, sr)
    y = normalize(y)
    y = trim_silence(y)
    y = pad_or_crop(y, target_len)
    return y, sr
