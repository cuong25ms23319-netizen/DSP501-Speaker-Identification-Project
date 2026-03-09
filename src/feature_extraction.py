"""
Module 4: feature_extraction.py
--------------------------------
Extract MFCC features from preprocessed audio.

Feature vector per file:
  - 13 MFCC coefficients × (mean + std) = shape (26,)
  - Captures vocal tract shape — highly speaker-discriminative

Output files:
  features/features_mfcc_raw.npy   → Pipeline A input
  features/features_mfcc_filt.npy  → Pipeline B input
  features/labels.npy              → integer speaker IDs
"""

import os
import numpy as np
import pandas as pd
import librosa

from preprocess import preprocess
from filter import design_fir, apply_filter


# ── MFCC parameters ────────────────────────────────────────────────
N_MFCC = 13
N_FFT = 512
HOP_LENGTH = 256
SR = 16000
TARGET_LEN = 48000   # 3 seconds × 16000 Hz


def extract_mfcc(y, sr=SR, n_mfcc=N_MFCC):
    """
    Extract MFCC feature vector from an audio array.

    Returns
    -------
    feature : 1-D array of shape (2 * n_mfcc,) = (mean, std) concatenated
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr,
                                 n_mfcc=n_mfcc,
                                 n_fft=N_FFT,
                                 hop_length=HOP_LENGTH)
    mean = mfcc.mean(axis=1)   # shape (n_mfcc,)
    std = mfcc.std(axis=1)     # shape (n_mfcc,)
    return np.concatenate([mean, std])


def build_dataset(index_csv, pipeline='raw', data_dir='data/raw'):
    """
    Build feature matrix X and label array y from an index CSV.

    Parameters
    ----------
    index_csv : path to CSV with columns [filename, speaker_id, speaker_name]
    pipeline  : 'raw' (Pipeline A) or 'filtered' (Pipeline B)
    data_dir  : root folder containing speaker subdirectories

    Returns
    -------
    X      : float array of shape (n_samples, 26)
    y      : int array of shape (n_samples,)   — speaker IDs
    names  : list of speaker name strings
    """
    df = pd.read_csv(index_csv)
    fir_coeffs = design_fir()  # only used for 'filtered' pipeline

    X, y, names = [], [], []

    for _, row in df.iterrows():
        path = os.path.join(data_dir, row['filename'])

        # Step 1: basic preprocessing (normalize, trim, pad)
        audio, sr = preprocess(path, sr=SR, target_len=TARGET_LEN)

        # Step 2: apply FIR filter for Pipeline B
        if pipeline == 'filtered':
            audio = apply_filter(audio, fir_coeffs)

        # Step 3: extract MFCC feature vector
        feat = extract_mfcc(audio, sr=sr)

        X.append(feat)
        y.append(int(row['speaker_id']))
        names.append(row['speaker_name'])

    return np.array(X), np.array(y), names


def save_features(index_csv, features_dir='features', data_dir='data/raw'):
    """
    Extract and save features for both pipelines.
    Call this once before training.
    """
    os.makedirs(features_dir, exist_ok=True)

    print("Extracting features — Pipeline A (raw) ...")
    X_raw, y, _ = build_dataset(index_csv, pipeline='raw', data_dir=data_dir)

    print("Extracting features — Pipeline B (filtered) ...")
    X_filt, _, _ = build_dataset(index_csv, pipeline='filtered', data_dir=data_dir)

    np.save(os.path.join(features_dir, 'features_mfcc_raw.npy'), X_raw)
    np.save(os.path.join(features_dir, 'features_mfcc_filt.npy'), X_filt)
    np.save(os.path.join(features_dir, 'labels.npy'), y)

    print(f"Saved: X_raw{X_raw.shape}, X_filt{X_filt.shape}, y{y.shape}")
    return X_raw, X_filt, y


# ── Quick test ──────────────────────────────────────────────────────
if __name__ == '__main__':
    # Resolve paths relative to project root (one level up from src/)
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_features(
        index_csv=os.path.join(ROOT, 'data', 'index.csv'),
        features_dir=os.path.join(ROOT, 'features'),
        data_dir=os.path.join(ROOT, 'data', 'raw'),
    )
