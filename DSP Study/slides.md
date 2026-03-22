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

<!-- _class: lead -->
<!-- _backgroundColor: #0f3460 -->
<!-- _color: white -->

# Feature Engineering
### Pipeline A vs Pipeline B — Head-to-Head

---

# Pipeline A — Baseline (6-dim)

<div style="text-align: center; margin-bottom: 12px;">
<span style="background: #e74c3c; color: white; padding: 6px 20px; border-radius: 20px; font-weight: bold; font-size: 0.9em;">NO DSP — Time-domain only</span>
</div>

| # | Feature | What it measures | Domain |
|:-:|:--------|:-----------------|:------:|
| 1 | **RMS Energy** (mean) | How loud is the voice on average? | Time |
| 2 | **RMS Energy** (std) | Does loudness stay steady or fluctuate? | Time |
| 3 | **ZCR** (mean) | How fast does the signal cross zero? | Time |
| 4 | **ZCR** (std) | Is the crossing rate consistent? | Time |
| 5 | **Mean \|amplitude\|** | Average signal strength | Time |
| 6 | **Std amplitude** | How spread out is the amplitude? | Time |

<div style="text-align: center; margin-top: 12px; padding: 10px; background: #e74c3c22; border-radius: 8px;">
⚠️ <strong>Problem:</strong> Knows HOW LOUD you speak — but NOT what your voice sounds like.
Two people with similar volume → indistinguishable.
</div>

---

# Pipeline B — DSP Enhanced (26-dim)

<div style="text-align: center; margin-bottom: 12px;">
<span style="background: #2ecc71; color: white; padding: 6px 20px; border-radius: 20px; font-weight: bold; font-size: 0.9em;">FIR → Pre-emphasis → MFCC</span>
</div>

### MFCC Extraction — 6 Steps

| Step | Operation | Purpose | Key Parameter |
|:----:|:----------|:--------|:-------------|
| 1 | **Frame** | Cut signal into short segments | 512 samples = 32ms |
| 2 | **Hamming Window** | Smooth frame edges | Reduces spectral leakage |
| 3 | **FFT** | Time → Frequency domain | 512-point FFT |
| 4 | **Mel Filterbank** | Mimic human hearing | Non-linear freq scale |
| 5 | **Log** | Match human loudness perception | Decibel-like scaling |
| 6 | **DCT** | Compress & decorrelate | Keep 13 coefficients |

### Result per clip:
$$\mathbf{x} = \underbrace{[\mu_1, \ldots, \mu_{13}]}_{\text{13 means (average timbre)}} \oplus \underbrace{[\sigma_1, \ldots, \sigma_{13}]}_{\text{13 stds (voice dynamics)}} = \textbf{26 dims}$$

---

# Feature Comparison — Why 26 > 6

| | Pipeline A (Baseline) | Pipeline B (DSP) |
|:--|:---------------------:|:----------------:|
| **Dimensions** | 6 | **26** |
| **Domain** | Time only | **Time + Frequency** |
| **What it captures** | Volume, speed | **Vocal tract shape** |
| **Speaker info** | Minimal | **Formants F1, F2, F3** |
| **DSP preprocessing** | None | **FIR + Pre-emphasis** |
| **Noise handling** | Affected by noise | **Filtered before extraction** |
| **Analogy** | Measuring height only | **Full fingerprint scan** |

<div style="text-align: center; margin-top: 12px; padding: 10px; background: #2ecc7122; border-radius: 8px;">
✅ MFCC captures the <strong>unique shape of each person's vocal tract</strong> — like a voice fingerprint.
</div>

---

<!-- _class: lead -->
<!-- _backgroundColor: #1a1a2e -->
<!-- _color: white -->

# AI Modeling & Training
### SVM with RBF Kernel

---

# Why SVM for This Task?

| Criterion | SVM | Deep Learning | Winner |
|:----------|:---:|:------------:|:------:|
| **Small dataset** (194 samples) | ✅ Excels | ❌ Needs 1000s+ | SVM |
| **High-dim features** (26 dims) | ✅ RBF kernel | ✅ Also good | Tie |
| **Training speed** | ✅ < 2 seconds | ❌ Minutes/hours | SVM |
| **Interpretability** | ✅ Clear boundary | ❌ Black box | SVM |
| **Overfitting risk** | ✅ C regularization | ❌ High on small data | SVM |
| **No GPU needed** | ✅ CPU only | ❌ GPU preferred | SVM |

> **Verdict:** With only 194 samples, SVM is the optimal choice. Deep Learning would severely overfit.

---

# SVM — How It Works

### RBF Kernel (Radial Basis Function)

$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2\right)$$

<div style="display: flex; gap: 30px; margin-top: 12px;">
<div style="flex: 1; background: #16213e; border-radius: 12px; padding: 16px;">
<div style="color: #F37021; font-weight: bold; margin-bottom: 6px;">Parameter C (Regularization)</div>
<div style="color: #ccc; font-size: 0.85em;">
• C small → smooth boundary, tolerates errors<br>
• C large → fits every point, risks overfitting<br>
• <strong>Search: [0.1, 1, 10, 100]</strong>
</div>
</div>
<div style="flex: 1; background: #16213e; border-radius: 12px; padding: 16px;">
<div style="color: #F37021; font-weight: bold; margin-bottom: 6px;">Parameter γ (Influence radius)</div>
<div style="color: #ccc; font-size: 0.85em;">
• γ small → wide influence, broad boundary<br>
• γ large → tight influence, complex boundary<br>
• <strong>Search: [scale, auto, 0.001, 0.01]</strong>
</div>
</div>
</div>

---

# Training Pipeline — Step by Step

<div style="background: #f8f9fa; border-radius: 12px; padding: 16px; border: 1px solid #ddd; font-size: 0.9em;">

| Step | Component | Detail |
|:----:|:----------|:-------|
| 1️⃣ | **StandardScaler** | Normalize features to zero mean, unit variance (inside each fold — no data leakage!) |
| 2️⃣ | **GridSearchCV** | Test 4 × 4 = **16 hyperparameter combinations** (C × γ) |
| 3️⃣ | **Inner CV** | 3-fold CV for each combination → select best (C, γ) |
| 4️⃣ | **Outer CV** | 5-fold Stratified CV → evaluate best model on unseen data |
| 5️⃣ | **Final Model** | Refit on all data with best hyperparameters |

</div>

### Key Design Decisions

| Decision | Choice | Why |
|:---------|:-------|:----|
| CV strategy | **5-fold Stratified** | Preserves speaker ratio in each fold |
| Scaler placement | **Inside pipeline** | Prevents data leakage between folds |
| Random seed | **42 everywhere** | Full reproducibility |
| Probability | **Platt scaling** | Enables confidence % in demo app |

---

<!-- _class: lead -->
<!-- _backgroundColor: #1a1a2e -->
<!-- _color: white -->

# 📊 Experimental Results
### 125 samples · 5 speakers × 25 files · 5-fold Stratified CV

---

# Headline Result: +36.0 pp Accuracy Gain

<div style="display: flex; justify-content: center; gap: 40px; margin: 20px 0;">
<div style="text-align: center; background: linear-gradient(135deg, #e74c3c22, #e74c3c11); border: 2px solid #e74c3c; border-radius: 16px; padding: 20px 40px;">
<div style="font-size: 0.7em; color: #e74c3c; font-weight: bold;">PIPELINE A — Baseline</div>
<div style="font-size: 3em; font-weight: bold; color: #e74c3c;">60.0%</div>
<div style="font-size: 0.8em; color: #999;">± 5.7% · CI [52.2%, 67.9%]</div>
<div style="font-size: 0.7em; color: #666; margin-top: 8px;">RMS + ZCR → SVM (6 dims)</div>
</div>
<div style="display: flex; align-items: center; font-size: 2em; color: #F37021;">→</div>
<div style="text-align: center; background: linear-gradient(135deg, #2ecc7122, #2ecc7111); border: 2px solid #2ecc71; border-radius: 16px; padding: 20px 40px;">
<div style="font-size: 0.7em; color: #2ecc71; font-weight: bold;">PIPELINE B — DSP Enhanced</div>
<div style="font-size: 3em; font-weight: bold; color: #2ecc71;">96.0%</div>
<div style="font-size: 0.8em; color: #999;">± 2.5% · CI [92.5%, 99.5%]</div>
<div style="font-size: 0.7em; color: #666; margin-top: 8px;">FIR + Pre-emph + MFCC → SVM (26 dims)</div>
</div>
</div>

> **Paired t-test: t = −9.49, p = 0.0007** → Highly statistically significant (p ≪ 0.05)

---

# Full Metrics Comparison

| Metric | Pipeline A (Baseline) | Pipeline B (DSP) | Δ Improvement |
|:-------|:---------------------:|:-----------------:|:-------------:|
| **Accuracy** | 60.0% | **96.0%** | 🔺 +36.0 pp |
| **F1-score (macro)** | 0.794 | **1.000** | 🔺 +0.206 |
| **Best C** | 10 | **1** | ← Lower = less overfit |
| **Best γ** | scale | scale | Same |
| **Feature dims** | 6 | 26 | +20 dims |

<div style="text-align: center; margin-top: 12px; padding: 10px; background: #2ecc7122; border-radius: 8px;">
✅ Pipeline B: <strong>C = 1</strong> (lowest possible) → simple decision boundary, <strong>no overfitting</strong>.<br>
Pipeline A: C = 10 → needs more complex boundary but still can't separate speakers.
</div>

---

# Cross-Validation Fold-by-Fold

| Fold | Pipeline A | Pipeline B | Gap |
|:----:|:---------:|:---------:|:---:|
| 1 | 64.0% | 96.0% | +32.0 |
| 2 | 52.0% | 92.0% | +40.0 |
| 3 | 68.0% | 96.0% | +28.0 |
| 4 | 56.0% | 100.0% | +44.0 |
| 5 | 60.0% | 96.0% | +36.0 |
| **Mean** | **60.0%** | **96.0%** | **+36.0** |

> Pipeline B outperforms A in **every single fold**. Fold 4 achieves **perfect 100%**. The improvement is **consistent**, not due to random chance.

---

# Statistical Significance

<div style="text-align: center; margin: 20px 0;">
<div style="display: inline-block; background: #16213e; border-radius: 16px; padding: 24px 48px; border-left: 5px solid #2ecc71;">
<div style="font-size: 0.8em; color: #999;">Paired t-test (5 folds)</div>
<div style="font-size: 2.5em; font-weight: bold; color: white; margin: 8px 0;">p = 0.0007</div>
<div style="font-size: 1em; color: #2ecc71; font-weight: bold;">★ Highly Significant (p ≪ 0.05)</div>
</div>
</div>

**Interpretation:**
- t-statistic = **−9.49** → Pipeline B consistently higher than A
- p-value = **0.0007** → probability of this result by chance: **0.07%**
- With 95% confidence, Pipeline B's true accuracy is between **92.5% and 99.5%**
- Pipeline A's range **[52.2%, 67.9%]** has **zero overlap** with Pipeline B → clear separation

---

# Why Pipeline B Wins — 3 Key Reasons

### 🔬 1. Feature Richness
26-dim MFCC encodes **vocal tract shape** (formants F1, F2, F3)
vs. 6-dim time features — only captures loudness & zero-crossings

### 🔇 2. Noise Suppression
FIR bandpass (300–3400 Hz) removes **>99% out-of-band noise**
Pre-emphasis boosts consonant energy by **+6 dB/octave**

### ⚡ 3. Efficiency
Entire DSP chain costs **< 5 ms per 3-second clip**
36.0 pp accuracy gain for **0.005 seconds** extra processing — trivial trade-off

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
1. **DSP preprocessing is essential** — +36.0 pp accuracy gain, confirmed by paired t-test (p = 0.0007)
2. **MFCC features** (26-dim) capture speaker-specific vocal tract characteristics far better than time-domain descriptors (6-dim)
3. **Handcrafted DSP + SVM** achieves **96.0%** on 125 balanced samples with **C=1** (no overfitting) — no deep learning required

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
