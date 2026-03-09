"""
Module 2: filter.py
-------------------
Design and apply a FIR bandpass filter for Pipeline B.

Why FIR (not IIR)?
  - Linear phase → no phase distortion on MFCC features
  - Always stable
  - Symmetric coefficients with odd numtaps → guaranteed linear phase

Passband: 300–3400 Hz  (ITU-T telephone speech band)
Window:   Hamming      (side lobes < -40 dB, good leakage control)
"""

import numpy as np
from scipy.signal import firwin, lfilter, freqz
import matplotlib.pyplot as plt


def design_fir(lowcut=300, highcut=3400, sr=16000, numtaps=101):
    """
    Design a FIR bandpass filter using the window method.

    Parameters
    ----------
    lowcut   : lower cutoff frequency in Hz
    highcut  : upper cutoff frequency in Hz
    sr       : sample rate in Hz
    numtaps  : number of filter taps (must be odd for linear phase)

    Returns
    -------
    coeffs : 1-D numpy array of FIR filter coefficients
    """
    nyq = sr / 2.0
    low = lowcut / nyq
    high = highcut / nyq
    coeffs = firwin(numtaps, [low, high], pass_zero=False, window='hamming')
    return coeffs


def apply_filter(y, coeffs):
    """Apply FIR filter coefficients to signal y using linear convolution."""
    return lfilter(coeffs, 1.0, y)


def plot_frequency_response(coeffs, sr=16000, save_path=None):
    """Plot the magnitude frequency response of the FIR filter."""
    w, h = freqz(coeffs, worN=2048)
    freqs = w * sr / (2 * np.pi)

    plt.figure(figsize=(8, 4))
    plt.plot(freqs, 20 * np.log10(np.abs(h) + 1e-10))
    plt.axvline(300, color='r', linestyle='--', label='300 Hz')
    plt.axvline(3400, color='g', linestyle='--', label='3400 Hz')
    plt.title('FIR Bandpass Filter — Frequency Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.xlim(0, sr / 2)
    plt.ylim(-80, 5)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_phase_response(coeffs, sr=16000, save_path=None):
    """Plot the phase frequency response of the FIR filter."""
    w, h = freqz(coeffs, worN=2048)
    freqs = w * sr / (2 * np.pi)

    plt.figure(figsize=(8, 4))
    plt.plot(freqs, np.unwrap(np.angle(h)))
    plt.axvline(300, color='r', linestyle='--', label='300 Hz')
    plt.axvline(3400, color='g', linestyle='--', label='3400 Hz')
    plt.title('FIR Bandpass Filter — Phase Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (radians)')
    plt.xlim(0, sr / 2)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
