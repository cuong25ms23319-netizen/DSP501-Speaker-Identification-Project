# Speaker Identification System
## Project Specification — DSP501, FPT University

---

## 1. Project Summary

Build a **closed-set Speaker Identification system** that recognizes the identity of a speaker from a `.wav` audio file. The system compares two pipelines — one with DSP preprocessing and one without — to quantitatively prove that DSP improves ML model performance.

**Core research question:** Does DSP preprocessing (FIR bandpass filtering + pre-emphasis) improve speaker identification accuracy?

---

## 2. Constraints & Requirements

| Item | Detail |
|---|---|
| Language | Python |
| ML approach | Classical ML only (no deep learning) |
| Models | SVM only (RBF kernel) |
| Pipelines | 2 (Pipeline A: basic time-domain, Pipeline B: DSP + MFCC) |
| Evaluation | Block split (70/30) + RepeatedStratifiedKFold (5-fold × 10 repeats), mean ± 95% CI, paired t-test |
| Dataset | Self-recorded, 4 speakers, 25 files each (100 total) |
| Audio format | `.wav`, 16 kHz, mono, 3 seconds |
| Demo | Streamlit web app |
| Report | IEEE format, 10–12 pages, min 10 references |

---

## 3. System Architecture

### Pipeline A — Baseline (no DSP, no frequency analysis)
```
Raw .wav
  → Preprocess (normalize, trim silence, pad/crop to 3s)
  → Extract basic time-domain features (RMS, ZCR, amplitude stats)
  → StandardScaler → SVM (RBF)
  → Predicted speaker
```
**Output:** 6-dimensional feature vector per sample

### Pipeline B — DSP Enhanced
```
Raw .wav
  → Preprocess (normalize, trim silence, pad/crop to 3s)
  → FIR Bandpass Filter (300–3400 Hz)
  → Pre-emphasis (α = 0.97)
  → Extract MFCC (mean + std)
  → StandardScaler → SVM (RBF)
  → Predicted speaker
```
**Output:** 26-dimensional feature vector per sample

### 2 Experiments

| ID | Pipeline | Model | Features | Dims |
|---|---|---|---|---|
| A1_SVM_basic | Raw → basic features | SVM | RMS, ZCR, amplitude | 6 |
| B1_SVM_dsp | FIR + Pre-emphasis → MFCC | SVM | MFCC mean+std | 26 |

---

## 4. Module Breakdown

### Module 1 — `preprocess.py`
**Purpose:** Load and standardize all audio files.

**Functions:**
- `load_audio(path, sr=16000)` → load mono, resample
- `normalize(y)` → amplitude normalize to peak 1.0
- `trim_silence(y, top_db=20)` → remove leading/trailing silence
- `pad_or_crop(y, target_len=48000)` → fixed 3s length
- `preprocess(path, sr, target_len)` → full pipeline combining all steps

**Output:** Preprocessed numpy array per file.

---

### Module 2 — `filter.py`
**Purpose:** Design and apply FIR bandpass filter.

**Functions:**
- `design_fir(lowcut=300, highcut=3400, sr=16000, numtaps=101)` → returns filter coefficients
- `apply_filter(y, coeffs)` → returns filtered signal via `lfilter`
- `plot_frequency_response(coeffs, sr, save_path)` → magnitude response plot
- `plot_phase_response(coeffs, sr, save_path)` → phase response plot

**Filter spec:**
- Type: FIR (not IIR — reason: linear phase, always stable)
- Window: Hamming
- Passband: 300–3400 Hz (telephone speech band, ITU-T standard)
- numtaps: 101 (odd, ensures symmetric → linear phase)

---

### Module 3 — `preemphasis.py`
**Purpose:** First-order pre-emphasis filter to flatten speech spectrum before MFCC extraction.

**Functions:**
- `pre_emphasize(y, alpha=0.97)` → y'[n] = y[n] − α·y[n−1]

**Why:** Speech energy drops ~6 dB/octave at high frequencies. Pre-emphasis compensates, improving MFCC discrimination.

**Applied:** After FIR filter, before MFCC extraction (Pipeline B only).

---

### Module 4 — `analysis.py`
**Purpose:** Frequency domain analysis for report figures.

**Functions:**
- `plot_waveform(y_raw, y_filt, sr, save_path)` → side-by-side waveform
- `plot_spectrum(y_raw, y_filt, sr, save_path)` → FFT magnitude spectrum
- `plot_stft(y_raw, y_filt, sr, save_path)` → STFT spectrogram comparison
- `compute_psd(y, sr)` → Power Spectral Density
- `compute_snr(y_raw, y_filt)` → SNR before/after filter

---

### Module 5 — `feature_extraction.py`
**Purpose:** Extract features for both pipelines.

**Functions:**
- `extract_basic_features(y, sr)` → Pipeline A: 6-dim time-domain vector
  - RMS energy (mean, std)
  - Zero Crossing Rate (mean, std)
  - Mean absolute amplitude
  - Std amplitude
- `extract_mfcc(y, sr, n_mfcc=13)` → Pipeline B: 26-dim MFCC vector (mean + std concatenated)
- `build_dataset(index_csv, pipeline='raw'|'filtered', data_dir)` → returns X, y arrays
- `save_features(index_csv, features_dir, data_dir)` → saves .npy files

**MFCC parameters:**
- `n_mfcc = 13`
- `n_fft = 512`
- `hop_length = 256`
- Aggregate: `np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])`

**Output files:**
- `features_basic.npy` → Pipeline A input (shape: 100 × 6)
- `features_mfcc_filt.npy` → Pipeline B input (shape: 100 × 26)
- `labels.npy` → speaker IDs (shape: 100)

---

### Module 6 — `train.py`
**Purpose:** Train and evaluate SVM models for both pipelines.

**Functions:**
- `train_svm(X, y)` → GridSearchCV, 5-fold CV, returns best model + metrics
- `run_experiment(name, X, y)` → runs full pipeline, saves results
- `save_results(results, path='results.json')`

**Hyperparameter search space:**
```python
SVM:
  C     : [0.01, 0.1, 1, 10]
  gamma : ['scale', 'auto', 0.001, 0.01]
  kernel: 'rbf'
```

**Training config:**
- Split: Block split 70/30 (file đầu → train, file cuối → test) to avoid data leakage
- CV: `RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)` — 50 evaluations
- GridSearchCV: `cv=5`
- Scaler: `StandardScaler` (inside Pipeline for SVM)
- Random seed: 42 everywhere

**Why block split?** Audio samples from the same speaker have correlation > 0.98 (same recording session). Random split causes data leakage → inflated accuracy.

---

### Module 7 — `evaluation.py`
**Purpose:** Compute all metrics and generate comparison figures.

**Functions:**
- `compute_metrics(y_true, y_pred)` → Accuracy, Precision, Recall, F1 macro
- `compute_ci(scores, confidence=0.95)` → 95% Confidence Interval (t-distribution)
- `paired_ttest(scores_a, scores_b)` → t-statistic, p-value
- `plot_confusion_matrix(y_true, y_pred, labels, title, save_path)` → seaborn heatmap
- `plot_roc_curve(model, X, y)` → one-vs-rest ROC
- `plot_comparison_table(results)` → summary bar chart

---

### Module 8 — `app.py` (Streamlit Demo)
**Purpose:** Smart Meeting Room demo — identify who is speaking + data collection.

**UI Tabs:**
1. **Test** — Identify speaker from mic recording or uploaded file
   - Shows Pipeline A vs Pipeline B predictions side-by-side
   - Confidence bars from `model.predict_proba()`
   - Waveform + MFCC heatmap visualizations (raw vs filtered)

2. **Thu âm (Recording)** — Record audio per standardized transcript
   - 25 standard Vietnamese sentences for consistent data collection
   - Tracks progress per speaker (need 25 samples minimum)
   - Saves to `data/raw/speaker_XX/` with auto CSV indexing

3. **Upload** — Batch upload audio files
   - Supports: wav, mp3, m4a, flac, ogg
   - Auto-converts to 16 kHz mono
   - Splits long files into 3-second chunks

4. **Quản lý (Management)** — Data management UI
   - View all speakers and file counts
   - Delete individual speakers or all data

5. **Train** — Model training interface
   - Data overview chart (files per speaker)
   - Requires ≥2 speakers to enable training
   - Runs feature extraction → training via subprocess
   - Displays results from `results.json`

**Git integration:** Teammates record locally, push to GitHub → app auto-detects new files on pull.

---

## 5. Dataset

### Speakers

| Folder | speaker_id | Name | Files |
|---|---|---|---|
| `speaker_08` | 7 | Cuong | 25 |
| `speaker_09` | 8 | Quang | 25 |
| `speaker_10` | 9 | Anne | 25 |
| `speaker_11` | 10 | Khoa | 25 |

**Total:** 4 speakers × 25 files = 100 samples
**Audio:** 16 kHz, mono, 3 seconds each

### Data leakage note
All 25 files per speaker are from the same recording session → within-speaker MFCC correlation > 0.98. Block split (temporal separation) is required to avoid leakage.

---

## 6. Folder Structure

```
dsp/
├── data/
│   ├── raw/                  # original .wav files
│   │   ├── speaker_08/       # Cuong (25 files)
│   │   ├── speaker_09/       # Quang (25 files)
│   │   ├── speaker_10/       # Anne (25 files)
│   │   └── speaker_11/       # Khoa (25 files)
│   └── index.csv             # filename, speaker_id, speaker_name
├── features/
│   ├── features_basic.npy    # Pipeline A (100 × 6)
│   ├── features_mfcc_filt.npy # Pipeline B (100 × 26)
│   └── labels.npy
├── models/
│   ├── svm_pipeline_a.pkl
│   └── svm_pipeline_b.pkl
├── figures/
│   ├── freq_response.png
│   ├── phase_response.png
│   ├── waveform_comparison.png
│   ├── spectrum_comparison.png
│   ├── stft_comparison.png
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   └── comparison_bar.png
├── notebooks/
│   ├── 01_analysis.ipynb           # Frequency-domain analysis
│   ├── 02_features.ipynb           # Feature extraction exploration
│   ├── 03_listen_processing.ipynb  # Audio preprocessing demos
│   ├── 03_train.ipynb              # Model training
│   ├── 04_evaluation.ipynb         # Results evaluation
│   ├── 04_f0_formant_analysis.ipynb # F0 and formant analysis
│   ├── 05_DSP_pipeline.ipynb       # DSP pipeline overview
│   ├── 05_speech_dsp_pipeline.ipynb # Speech DSP detailed pipeline
│   ├── 06_evaluation.ipynb         # Additional evaluation
│   └── final_dsp_pipeline_analysis.ipynb # Final comprehensive analysis
├── src/
│   ├── preprocess.py
│   ├── filter.py
│   ├── preemphasis.py
│   ├── analysis.py
│   ├── feature_extraction.py
│   ├── train.py
│   └── evaluation.py
├── app.py
├── results.json
├── requirements.txt
└── README.md
```

---

## 7. Results Schema (`results.json`)

```json
{
  "random_seed": 42,
  "cv_folds": 5,
  "experiments": {
    "A1_SVM_basic": {
      "best_params": {"C": 1, "gamma": "scale"},
      "accuracy":  {"mean": 0.59, "std": 0.073, "ci_95": [0.488, 0.692]},
      "f1_macro":  {"mean": 0.704, "std": 0.0, "ci_95": [0.0, 0.0]},
      "precision": {"mean": 0.0, "std": 0.0},
      "recall":    {"mean": 0.0, "std": 0.0}
    },
    "B1_SVM_dsp": {
      "best_params": {"C": 1, "gamma": "scale"},
      "accuracy":  {"mean": 0.97, "std": 0.040, "ci_95": [0.914, 1.026]},
      "f1_macro":  {"mean": 1.0, "std": 0.0, "ci_95": [0.0, 0.0]},
      "precision": {"mean": 0.0, "std": 0.0},
      "recall":    {"mean": 0.0, "std": 0.0}
    }
  },
  "statistical_tests": {
    "SVM_A_vs_B": {"t_stat": -8.718, "p_value": 0.00095}
  }
}
```

---

## 8. Key Technical Decisions & Justifications

| Decision | Choice | Reason |
|---|---|---|
| Filter type | FIR | Linear phase — no phase distortion on MFCC |
| Passband | 300–3400 Hz | ITU-T telephone band, core speech energy |
| Window | Hamming | Side lobes < −40 dB, good spectral leakage control |
| Pre-emphasis | α = 0.97 | Flattens speech spectrum, improves MFCC discrimination |
| Pipeline A features | RMS, ZCR, amplitude (6-dim) | Time-domain only baseline — no frequency analysis |
| Pipeline B features | MFCC mean+std (26-dim) | Captures vocal tract shape, speaker-discriminative |
| n_mfcc | 13 | Captures fundamental + first formants, standard |
| Scaler | StandardScaler | SVM sensitive to feature scale |
| Train/test split | Block split 70/30 | Avoids data leakage from same-session recordings (corr > 0.98) |
| CV strategy | RepeatedStratifiedKFold 5×10 | 50 evaluations — robust with small dataset (100 samples) |
| C range | [0.01, 0.1, 1, 10] | Lower max C to prevent overfitting on small dataset |
| Aggregation | mean + std | Temporal summary, fixed-length vector for SVM |

---

## 9. Report Structure (IEEE format, 10–12 pages)

1. Introduction — problem, motivation, use case, research question
2. Signal Analysis — dataset description, raw signal characteristics
3. DSP Methodology — FIR design, pre-emphasis, freq/phase response, FFT/STFT/PSD, SNR
4. Feature Engineering — basic features (Pipeline A) vs MFCC (Pipeline B), parameter justification
5. AI Modeling — SVM, hyperparameter tuning, CV strategy, block split rationale
6. Experimental Results — metrics table, confusion matrix, ROC
7. Comparative Analysis — Pipeline A vs B (DSP impact), data leakage discussion
8. Discussion — 6 required questions
9. Limitations — dataset size (100 samples), same-session bias, 4 speakers only
10. Conclusion
11. Ethics Statement — consent, bias, reproducibility
12. References — minimum 10 academic sources

---

## 10. Discussion Questions (must answer in report)

1. Does DSP preprocessing improve performance? Why or why not?
2. Which frequency bands contain the most discriminative information?
3. Does the filter remove any useful information?
4. How does preprocessing affect overfitting?
5. What is the computational complexity trade-off between Pipeline A and B?
6. Would DSP still be necessary if using deep learning?

---

## 11. Evaluation Criteria

| Criterion | Weight |
|---|---|
| DSP Design & Theoretical Justification | 25% |
| Before vs After DSP Comparison (A1 vs B1) | 20% |
| AI/ML Model Design (SVM) | 20% |
| Experimental Rigor | 15% |
| Critical Discussion | 10% |
| Report Quality | 5% |
| Innovation | 5% |
