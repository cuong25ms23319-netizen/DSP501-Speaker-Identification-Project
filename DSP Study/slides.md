---
marp: true
theme: default
paginate: true
backgroundColor: #fff
color: #333
style: |
  section {
    font-family: 'Segoe UI', Arial, sans-serif;
  }
  h1, h2 {
    color: #F37021;
  }
  table {
    font-size: 0.85em;
  }
  blockquote {
    border-left: 4px solid #F37021;
    padding-left: 1em;
    color: #555;
  }
---

<!-- _class: lead -->
<!-- _backgroundColor: #F37021 -->
<!-- _color: white -->

# Speaker Identification System
## DSP-Enhanced Pipeline vs. Baseline

**Nguyen Huy Cuong | Hon Vi Dan | Le Nhut Thanh Quang | Nguyen Duc Minh Khoa**

FPT University — DSP501 | Supervisor: Dr. Dang Ngoc Minh Duc

---

# Outline

1. Introduction
2. Dataset
3. DSP Methodology
4. Feature Engineering
5. AI Modeling
6. Experimental Results
7. Discussion
8. Conclusion
9. Live Demo

---

# Problem & Motivation

- **Goal:** Identify *who* is speaking from a short audio clip
- **Challenge:** Speech is non-stationary, noisy, varies across sessions
- **Research question:** Does DSP preprocessing improve speaker identification accuracy?

### Approach: Compare two pipelines

| Pipeline | Flow |
|----------|------|
| **A (Baseline)** | Raw signal → basic time features → SVM |
| **B (DSP-enhanced)** | FIR filter + pre-emphasis → MFCC → SVM |

---

# Dataset

### 5 Speakers × 25 files = 125 samples

| Speaker | ID | Files |
|---------|----|-------|
| Dan | 07 | 25 |
| Cuong | 08 | 25 |
| Quang | 09 | 25 |
| Anne | 10 | 25 |
| Khoa | 11 | 25 |

- Duration: ~3 seconds each
- Mono WAV, fs = 16,000 Hz → 48,000 samples/clip

### Preprocessing: Load → Normalize [-1,1] → Trim silence → Pad/crop to 48,000

---

# FIR Bandpass Filter (300–3400 Hz)

### Purpose
- Isolate speech formants (F1, F2, F3)
- Remove low-frequency hum (<300 Hz) and high-frequency noise (>3400 Hz)

### Design
- **Method:** Window method (Hamming), 101 taps
- **Ideal impulse response:**

$$h_{ideal}[n] = \frac{\sin(\omega_{high}(n - M/2))}{\pi(n - M/2)} - \frac{\sin(\omega_{low}(n - M/2))}{\pi(n - M/2)}$$

- **Final coefficients:** h[n] = h_ideal[n] × w[n]

---

# Pre-emphasis Filter

### Problem
Speech energy drops **-6 dB/octave** → high frequencies (consonants) are weak

### Solution: First-order high-pass filter

$$y'[n] = y[n] - 0.97 \cdot y[n-1]$$

### Effect
- Boosts high-frequency consonant/sibilant energy
- Flattens spectrum → better MFCC extraction
- Balances formant energies across frequency range

---

# Pipeline A — Baseline Features (6-dim)

| # | Feature | Meaning |
|---|---------|---------|
| 1 | RMS Energy (mean) | Average loudness |
| 2 | RMS Energy (std) | Energy variation |
| 3 | ZCR (mean) | Signal sign-change rate |
| 4 | ZCR (std) | ZCR variation across frames |
| 5 | Mean \|amplitude\| | Average absolute amplitude |
| 6 | Std amplitude | Amplitude dispersion |

> **Only time-domain information — no frequency content!**

---

# Pipeline B — MFCC Extraction (26-dim)

### 6-step process
1. **Frame** signal into 512-sample windows (hop = 256)
2. **Hamming window** each frame → reduce spectral leakage
3. **FFT** → power spectrum
4. **Mel filterbank** (mimics human hearing)
5. **Log** of filterbank energies
6. **DCT** → 13 MFCCs per frame

### Final vector per clip
$$\mathbf{x} = [\mu_1, ..., \mu_{13},\ \sigma_1, ..., \sigma_{13}]$$

Mean + Std of 13 MFCCs = **26 dimensions**

---

# SVM with RBF Kernel

### Why SVM?
- Works well with small datasets (125 samples)
- Effective in high-dimensional space

### RBF Kernel
$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$$

### Hyperparameter search (GridSearchCV, 3-fold inner)

| Parameter | Values |
|-----------|--------|
| C | 0.1, 1, 10, 100 |
| gamma | scale, auto, 0.001, 0.01 |

**Evaluation:** 5-fold Stratified Cross-Validation (seed = 42)

---

<!-- _class: lead -->
<!-- _backgroundColor: #F37021 -->
<!-- _color: white -->

# Experimental Results

---

# Results

| Pipeline | Best C | Best gamma | Mean Accuracy | 95% CI |
|----------|--------|-----------|---------------|--------|
| **A — Baseline** | 1 | scale | 56.3% ± 2.8% | [52.4%, 60.1%] |
| **B — DSP-enhanced** | 1 | scale | **97.0% ± 3.6%** | [92.0%, 102.1%] |

### F1-score (Macro)
- Pipeline A: **0.690**
- Pipeline B: **1.000**

### Paired t-test
$$t = -34.79, \quad p = 0.000004$$

> p << 0.05 → **Highly statistically significant!**

---

# Why Pipeline B Wins

### 1. Feature Richness
26-dim MFCCs capture vocal tract shape
vs. 6-dim time features — only amplitude info

### 2. Noise Suppression
FIR bandpass + pre-emphasis isolate speech frequencies
→ improved SNR before feature extraction

### 3. Massive Accuracy Gap
**+40.7 percentage points** (56.3% → 97.0%)
Computational cost: **<1 ms** per 3-second clip

---

# Limitations

- **Small dataset:** 5 speakers, 125 samples — may not generalize
- **Clean recordings only:** Trained in quiet; noisy environments (classroom) degrade mic predictions
- **Closed-set:** Cannot reject unknown speakers
- **Single classifier:** Only SVM tested
- **Fixed filter:** 300–3400 Hz not optimized per dataset

---

# Conclusion

### Key Findings
1. **DSP preprocessing is essential** — 40.7 pp accuracy gain, confirmed by paired t-test (p < 0.001)
2. **MFCC features** capture speaker-specific vocal tract characteristics far better than time-domain descriptors
3. **Handcrafted DSP + SVM** achieves 97% on small dataset — no deep learning required

### Future Work
- More speakers, noisy conditions
- Noise reduction for real-time mic input
- Integrate DSP front-end with neural networks

---

<!-- _class: lead -->
<!-- _backgroundColor: #F37021 -->
<!-- _color: white -->

# Live Demo
### Streamlit App — Speaker Identification
Upload WAV file or record from mic

---

<!-- _class: lead -->
<!-- _backgroundColor: #F37021 -->
<!-- _color: white -->

# Thank You!
### Questions?
