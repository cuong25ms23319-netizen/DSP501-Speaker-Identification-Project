"""
Module 3: analysis.py
---------------------
Frequency-domain analysis and visualization for the report.

Generates figures comparing raw vs filtered signals:
  - Waveform
  - FFT magnitude spectrum
  - STFT spectrogram
  - Power Spectral Density (PSD)
  - Signal-to-Noise Ratio (SNR)
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.signal import welch


def plot_waveform(y_raw, y_filt, sr, save_path=None):
    """Side-by-side waveform: raw vs filtered."""
    time = np.linspace(0, len(y_raw) / sr, len(y_raw))

    fig, axes = plt.subplots(1, 2, figsize=(12, 3), sharey=True)
    axes[0].plot(time, y_raw, linewidth=0.5)
    axes[0].set_title('Raw Signal')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')

    axes[1].plot(time, y_filt, linewidth=0.5, color='orange')
    axes[1].set_title('Filtered Signal (300–3400 Hz)')
    axes[1].set_xlabel('Time (s)')

    plt.suptitle('Waveform Comparison', fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_spectrum(y_raw, y_filt, sr, save_path=None):
    """FFT magnitude spectrum: raw vs filtered."""
    def fft_magnitude(y):
        n = len(y)
        Y = np.abs(np.fft.rfft(y)) / n
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        return freqs, Y

    freqs_r, mag_r = fft_magnitude(y_raw)
    freqs_f, mag_f = fft_magnitude(y_filt)

    plt.figure(figsize=(10, 4))
    plt.plot(freqs_r, 20 * np.log10(mag_r + 1e-10), label='Raw', alpha=0.7)
    plt.plot(freqs_f, 20 * np.log10(mag_f + 1e-10), label='Filtered', alpha=0.7, color='orange')
    plt.axvline(300, color='r', linestyle='--', linewidth=0.8, label='300 Hz')
    plt.axvline(3400, color='g', linestyle='--', linewidth=0.8, label='3400 Hz')
    plt.title('FFT Magnitude Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.xlim(0, sr / 2)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_stft(y_raw, y_filt, sr, save_path=None):
    """STFT spectrogram comparison: raw vs filtered."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, y, title in zip(axes, [y_raw, y_filt], ['Raw', 'Filtered']):
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, hop_length=256,
                                       x_axis='time', y_axis='hz', ax=ax)
        ax.set_title(f'STFT — {title}')
        fig.colorbar(img, ax=ax, format='%+2.0f dB')

    plt.suptitle('Spectrogram Comparison', fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def compute_psd(y, sr):
    """
    Compute Power Spectral Density using Welch's method.

    Returns
    -------
    freqs : frequency array (Hz)
    psd   : power values
    """
    freqs, psd = welch(y, fs=sr, nperseg=512)
    return freqs, psd


def compute_snr(y_raw, y_filt):
    """
    Estimate SNR improvement after filtering.

    SNR = 10 * log10(signal_power / noise_power)
    noise ≈ difference between raw and filtered
    """
    noise = y_raw - y_filt
    signal_power = np.mean(y_filt ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return float('inf')
    snr = 10 * np.log10(signal_power / noise_power)
    return snr
