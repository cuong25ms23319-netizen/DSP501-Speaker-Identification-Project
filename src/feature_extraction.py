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

    Steps:
      1. Compute 13 MFCC coefficients per frame
      2. Drop MFCC[0] (log-energy — channel-dependent)
      3. Compute delta (velocity) and delta-delta (acceleration)
      4. Aggregate: mean + std of MFCC, delta, delta-delta → 72-dim vector

    Delta features capture HOW the voice changes over time —
    more robust to mic/channel differences than static MFCC alone.

    Returns
    -------
    feature : 1-D array of shape (72,)
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr,
                                 n_mfcc=n_mfcc,
                                 n_fft=N_FFT,
                                 hop_length=HOP_LENGTH)
    mfcc = mfcc[1:]            # drop MFCC[0] (energy — channel-dependent)

    # Delta (velocity) — how MFCC changes frame-to-frame
    delta = librosa.feature.delta(mfcc)
    # Delta-delta (acceleration) — rate of change of delta
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Aggregate each: mean + std across time
    parts = []
    for feat in [mfcc, delta, delta2]:
        parts.append(feat.mean(axis=1))   # (12,)
        parts.append(feat.std(axis=1))    # (12,)

    return np.concatenate(parts)  # 12*6 = 72 dims


def augment_audio(y, sr=SR):
    """
    Generate augmented versions of audio for robustness.
    Simulates mic/environment variations: volume change, noise, pitch shift.
    """
    augmented = []
    # 1. Volume variations (mic gain differences)
    for gain in [0.5, 0.7, 1.3]:
        augmented.append(np.clip(y * gain, -1.0, 1.0))
    # 2. Add background noise at various SNR levels
    for snr_db in [20, 15, 10]:
        noise = np.random.randn(len(y)) * np.sqrt(np.mean(y**2) / (10**(snr_db/10)))
        augmented.append(np.clip(y + noise, -1.0, 1.0))
    # 3. Slight pitch shift (simulates different mic frequency response)
    for n_steps in [-0.5, 0.5]:
        augmented.append(librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps))
    return augmented


def build_dataset(index_csv, pipeline='raw', data_dir='data/raw', augment=False):
    """
    Build feature matrix X and label array y from an index CSV.

    Parameters
    ----------
    index_csv : path to CSV with columns [filename, speaker_id, speaker_name]
    pipeline  : 'raw' (Pipeline A) or 'filtered' (Pipeline B)
    data_dir  : root folder containing speaker subdirectories
    augment   : if True, add augmented versions for Pipeline B

    Returns
    -------
    X      : float array
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
            audio_f = apply_filter(audio, fir_coeffs)
            audio_f = pre_emphasize(audio_f)
            feat = extract_mfcc(audio_f, sr=sr)
            X.append(feat)
            y.append(int(row['speaker_id']))
            names.append(row['speaker_name'])

            # Data augmentation: add variations for robustness
            if augment:
                for aug_audio in augment_audio(audio, sr):
                    aug_f = apply_filter(aug_audio, fir_coeffs)
                    aug_f = pre_emphasize(aug_f)
                    X.append(extract_mfcc(aug_f, sr=sr))
                    y.append(int(row['speaker_id']))
                    names.append(row['speaker_name'])
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
    X_basic, y_basic, _ = build_dataset(index_csv, pipeline='raw', data_dir=data_dir)

    print("Extracting features — Pipeline B (FIR + pre-emphasis + MFCC) + augmentation ...")
    X_mfcc, y_mfcc, _ = build_dataset(index_csv, pipeline='filtered', data_dir=data_dir, augment=True)

    np.save(os.path.join(features_dir, 'features_basic.npy'), X_basic)
    np.save(os.path.join(features_dir, 'features_mfcc_filt.npy'), X_mfcc)
    np.save(os.path.join(features_dir, 'labels_basic.npy'), y_basic)
    np.save(os.path.join(features_dir, 'labels_mfcc.npy'), y_mfcc)
    # Backward compat
    np.save(os.path.join(features_dir, 'labels.npy'), y_basic)

    print(f"Saved: X_basic{X_basic.shape}, X_mfcc{X_mfcc.shape}")
    print(f"       y_basic{y_basic.shape}, y_mfcc{y_mfcc.shape}")
    return X_basic, X_mfcc, y_basic, y_mfcc


# ── Quick test ──────────────────────────────────────────────────────
if __name__ == '__main__':
    # Resolve paths relative to project root (one level up from src/)
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_features(
        index_csv=os.path.join(ROOT, 'data', 'index.csv'),
        features_dir=os.path.join(ROOT, 'features'),
        data_dir=os.path.join(ROOT, 'data', 'raw'),
    )
