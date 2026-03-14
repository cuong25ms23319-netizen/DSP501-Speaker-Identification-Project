"""
Module 4: feature_extraction.py
--------------------------------
Feature extraction for both pipelines.

Pipeline A (Baseline — no DSP):
  Audio → basic time-domain features (energy, ZCR, stats)
  Output: features/features_basic.npy

Pipeline B (DSP Enhanced):
  Audio → FIR filter → pre-emphasis → MFCC (mean + std)
  Output: features/features_mfcc_filt.npy

Labels: features/labels.npy
"""

import os
import numpy as np
import pandas as pd
import librosa
from preprocess import preprocess
from filter import design_fir, apply_filter
from preemphasis import pre_emphasize


# ── Parameters ────────────────────────────────────────────────────
N_MFCC = 13
N_FFT = 512
HOP_LENGTH = 256
SR = 16000
TARGET_LEN = 48000   # 3 seconds × 16000 Hz


def extract_basic_features(y, sr=SR):
    """
    Pipeline A: basic time-domain features — no DSP, no frequency analysis.

    Features (6 dims):
      1. RMS energy (mean)
      2. RMS energy (std)
      3. Zero Crossing Rate (mean)
      4. Zero Crossing Rate (std)
      5. Mean absolute amplitude
      6. Std amplitude
    """
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]

    return np.array([
        rms.mean(),
        rms.std(),
        zcr.mean(),
        zcr.std(),
        np.mean(np.abs(y)),
        np.std(y),
    ], dtype=np.float64)


def extract_mfcc(y, sr=SR, n_mfcc=N_MFCC):
    """
    Pipeline B: MFCC feature vector (after FIR + pre-emphasis).

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
    X      : float array — shape (n, 10) for raw, (n, 26) for filtered
    y      : int array of shape (n_samples,)
    names  : list of speaker name strings
    """
    df = pd.read_csv(index_csv)
    fir_coeffs = design_fir()

    X, y, names = [], [], []

    for _, row in df.iterrows():
        path = os.path.join(data_dir, row['filename'])
        audio, sr = preprocess(path, sr=SR, target_len=TARGET_LEN)

        if pipeline == 'filtered':
            # Pipeline B: FIR → pre-emphasis → MFCC
            audio = apply_filter(audio, fir_coeffs)
            audio = pre_emphasize(audio)
            feat = extract_mfcc(audio, sr=sr)
        else:
            # Pipeline A: basic time-domain features only
            feat = extract_basic_features(audio, sr=sr)

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

    print("Extracting features — Pipeline A (basic time-domain) ...")
    X_basic, y, _ = build_dataset(index_csv, pipeline='raw', data_dir=data_dir)

    print("Extracting features — Pipeline B (FIR + pre-emphasis + MFCC) ...")
    X_mfcc, _, _ = build_dataset(index_csv, pipeline='filtered', data_dir=data_dir)

    np.save(os.path.join(features_dir, 'features_basic.npy'), X_basic)
    np.save(os.path.join(features_dir, 'features_mfcc_filt.npy'), X_mfcc)
    np.save(os.path.join(features_dir, 'labels.npy'), y)

    print(f"Saved: X_basic{X_basic.shape}, X_mfcc{X_mfcc.shape}, y{y.shape}")
    return X_basic, X_mfcc, y


# ── Quick test ──────────────────────────────────────────────────────
if __name__ == '__main__':
    # Resolve paths relative to project root (one level up from src/)
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_features(
        index_csv=os.path.join(ROOT, 'data', 'index.csv'),
        features_dir=os.path.join(ROOT, 'features'),
        data_dir=os.path.join(ROOT, 'data', 'raw'),
    )
