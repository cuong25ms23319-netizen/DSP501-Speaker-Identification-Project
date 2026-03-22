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


def sliding_windows(audio, window_len, hop_len):
    """
    Split audio into overlapping windows.
    Returns list of audio segments, each of length window_len.
    """
    segments = []
    for start in range(0, len(audio) - window_len + 1, hop_len):
        segments.append(audio[start:start + window_len])
    return segments


# Sliding window parameters
WINDOW_SEC = 1.5    # 1.5-second windows
HOP_SEC = 0.5       # 0.5-second hop → 4 windows per 3s clip
WINDOW_LEN = int(WINDOW_SEC * SR)  # 24000 samples
HOP_LEN_SW = int(HOP_SEC * SR)     # 8000 samples


def build_dataset(index_csv, pipeline='raw', data_dir='data/raw', use_sliding_window=False):
    """
    Build feature matrix X and label array y from an index CSV.

    Parameters
    ----------
    index_csv : path to CSV with columns [filename, speaker_id, speaker_name]
    pipeline  : 'raw' (Pipeline A) or 'filtered' (Pipeline B)
    data_dir  : root folder containing speaker subdirectories
    use_sliding_window : if True, split each clip into overlapping windows
                         to increase sample count with real data

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

        # Get audio segments (1 original or multiple windows)
        if use_sliding_window:
            segments = sliding_windows(audio, WINDOW_LEN, HOP_LEN_SW)
        else:
            segments = [audio]

        for seg in segments:
            if pipeline == 'filtered':
                # Pipeline B: FIR → pre-emphasis → MFCC
                seg_f = apply_filter(seg, fir_coeffs)
                seg_f = pre_emphasize(seg_f)
                feat = extract_mfcc(seg_f, sr=sr)
            else:
                # Pipeline A: basic time-domain features only
                feat = extract_basic_features(seg, sr=sr)

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
