# Speaker Identification System
## Project Specification — DSP501, FPT University

---

## 1. Project Summary

Build a **closed-set Speaker Identification system** that recognizes the identity of a speaker from a `.wav` audio file. The system must compare two pipelines — one with DSP preprocessing and one without — to quantitatively prove that DSP improves ML model performance.

**Core research question:** Does FIR bandpass filtering improve speaker identification accuracy?

---

## 2. Constraints & Requirements

| Item | Detail |
|---|---|
| Language | Python |
| ML approach | Classical ML only (no deep learning) |
| Models | SVM only |
| Pipelines | 2 (Pipeline A: raw, Pipeline B: filtered) |
| Evaluation | 5-fold stratified CV, mean ± 95% CI, paired t-test |
| Dataset | Public dataset or self-recorded (5–6 speakers, 10–12 files each) |
| Audio format | `.wav`, 16 kHz, mono, 3 seconds |
| Demo | Streamlit web app |
| Report | IEEE format, 10–12 pages, min 10 references |

---

## 3. System Architecture

### Pipeline A — Baseline (no DSP)
```
Raw .wav
  → Normalize amplitude
  → Extract MFCC (mean + std)
  → SVM / Random Forest
  → Predicted speaker
```

### Pipeline B — DSP Enhanced
```
Raw .wav
  → Normalize amplitude
  → FIR Bandpass Filter (300–3400 Hz)
  → Extract MFCC (mean + std)
  → SVM / Random Forest
  → Predicted speaker
```

### 2 Experiments

| ID | Pipeline | Model | Input |
|---|---|---|---|
| A1 | Raw | SVM | MFCC raw |
| B1 | Filtered | SVM | MFCC filtered |

---

## 4. Module Breakdown

### Module 1 — `preprocess.py`
**Purpose:** Load and standardize all audio files.

**Functions needed:**
- `load_audio(path, sr=16000)` → load mono, resample
- `normalize(y)` → amplitude normalize to [-1, 1]
- `trim_silence(y, top_db=20)` → remove leading/trailing silence
- `pad_or_crop(y, target_len=48000)` → fixed 3s length

**Output:** Preprocessed numpy array per file.

---

### Module 2 — `filter.py`
**Purpose:** Design and apply FIR bandpass filter.

**Functions needed:**
- `design_fir(lowcut=300, highcut=3400, sr=16000, numtaps=101)` → returns filter coefficients
- `apply_filter(y, coeffs)` → returns filtered signal
- `plot_frequency_response(coeffs, sr)` → magnitude response plot
- `plot_phase_response(coeffs, sr)` → phase response plot

**Filter spec:**
- Type: FIR (not IIR — reason: linear phase, always stable)
- Window: Hamming
- Passband: 300–3400 Hz (telephone speech band, ITU-T standard)
- numtaps: 101 (odd, ensures symmetric → linear phase)

---

### Module 3 — `analysis.py`
**Purpose:** Frequency domain analysis for report figures.

**Functions needed:**
- `plot_waveform(y_raw, y_filt, sr)` → side-by-side waveform
- `plot_spectrum(y_raw, y_filt, sr)` → FFT magnitude spectrum
- `plot_stft(y_raw, y_filt, sr)` → STFT spectrogram comparison
- `compute_psd(y, sr)` → Power Spectral Density
- `compute_snr(y_raw, y_filt)` → SNR before/after filter

---

### Module 4 — `feature_extraction.py`
**Purpose:** Extract MFCC features from audio.

**Functions needed:**
- `extract_mfcc(y, sr, n_mfcc=13)` → returns (mean+std) vector, shape (26,)
- `build_dataset(index_csv, pipeline='raw'|'filtered')` → returns X, y arrays

**MFCC parameters:**
- `n_mfcc = 13`
- `n_fft = 512`
- `hop_length = 256`
- Aggregate: `np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])`

**Output files:**
- `features_mfcc_raw.npy` → Pipeline A input
- `features_mfcc_filt.npy` → Pipeline B input
- `labels.npy`

---

### Module 5 — `train.py`
**Purpose:** Train and evaluate all 4 experiments.

**Functions needed:**
- `train_svm(X, y)` → GridSearchCV, 5-fold CV, returns best model + metrics
- `run_experiment(name, X, y)` → runs full pipeline, saves results
- `save_results(results, path='results.json')`

**Hyperparameter search space:**
```python
SVM:
  C     : [0.1, 1, 10, 100]
  gamma : ['scale', 'auto', 0.001, 0.01]
  kernel: 'rbf'
```

**Training config:**
- CV: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- Scaler: `StandardScaler` (inside Pipeline for SVM)
- Random seed: 42 everywhere

---

### Module 6 — `evaluation.py`
**Purpose:** Compute all metrics and generate comparison figures.

**Functions needed:**
- `compute_metrics(y_true, y_pred)` → Accuracy, Precision, Recall, F1 macro
- `compute_ci(scores, confidence=0.95)` → 95% Confidence Interval
- `paired_ttest(scores_a, scores_b)` → t-statistic, p-value
- `plot_confusion_matrix(y_true, y_pred, labels)` → seaborn heatmap
- `plot_roc_curve(model, X, y)` → one-vs-rest ROC
- `plot_comparison_table(results)` → summary bar chart

---

### Module 7 — `app.py` (Streamlit Demo)
**Purpose:** Smart Meeting Room demo — identify who is speaking.

**UI Flow:**
```
Sidebar: list of registered speakers (names + avatars)
Main:
  1. Upload .wav file (or record)
  2. Toggle: Pipeline A vs Pipeline B
  3. Button: "Identify Speaker"
  4. Result: Speaker name + confidence bar
  5. Visualization: waveform raw vs filtered + MFCC heatmap
  6. Side-by-side comparison: Pipeline A result vs Pipeline B result
```

**Key implementation notes:**
- Load pre-trained models with `joblib.load()`
- Apply same preprocessing + feature extraction as training
- Show confidence as `model.predict_proba()`
- Use `streamlit-audio-recorder` for live recording if needed

---

## 5. Folder Structure

```
speaker-identification/
├── data/
│   ├── raw/                  # original .wav files
│   │   ├── speaker_01/
│   │   └── speaker_02/
│   └── index.csv             # filename, speaker_id, speaker_name
├── features/
│   ├── features_mfcc_raw.npy
│   ├── features_mfcc_filt.npy
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
│   ├── confusion_matrix_*.png
│   └── roc_curve_*.png
├── notebooks/
│   ├── 01_analysis.ipynb
│   ├── 02_features.ipynb
│   ├── 03_train.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── preprocess.py
│   ├── filter.py
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

## 6. Results Schema (`results.json`)

```json
{
  "random_seed": 42,
  "cv_folds": 5,
  "experiments": {
    "A1_SVM_raw": {
      "best_params": {"C": 10, "gamma": "scale"},
      "accuracy":  {"mean": 0.0, "std": 0.0, "ci_95": [0.0, 0.0]},
      "f1_macro":  {"mean": 0.0, "std": 0.0, "ci_95": [0.0, 0.0]},
      "precision": {"mean": 0.0, "std": 0.0},
      "recall":    {"mean": 0.0, "std": 0.0}
    },
    "B1_SVM_filt": { ... }
  },
  "statistical_tests": {
    "SVM_A_vs_B": {"t_stat": 0.0, "p_value": 0.0}
  }
}
```

---

## 7. Key Technical Decisions & Justifications

| Decision | Choice | Reason |
|---|---|---|
| Filter type | FIR | Linear phase — no phase distortion on MFCC |
| Passband | 300–3400 Hz | ITU-T telephone band, core speech energy |
| Window | Hamming | Side lobes < −40 dB, good spectral leakage control |
| Feature | MFCC mean+std | Captures vocal tract shape, speaker-discriminative |
| n_mfcc | 13 | Captures fundamental + first formants, standard |
| Scaler | StandardScaler | SVM sensitive to feature scale |
| CV strategy | StratifiedKFold | Preserves class distribution per fold |
| Aggregation | mean + std | Temporal summary, fixed-length vector for SVM |

---

## 8. Report Structure (IEEE format, 10–12 pages)

1. Introduction — problem, motivation, use case, research question
2. Signal Analysis — dataset description, raw signal characteristics
3. DSP Methodology — FIR design, freq/phase response, FFT/STFT/PSD, SNR
4. Feature Engineering — MFCC math, parameter justification
5. AI Modeling — SVM + RF, hyperparameter tuning, CV strategy
6. Experimental Results — metrics table, confusion matrix, ROC
7. Comparative Analysis — A vs B (DSP impact), SVM vs RF
8. Discussion — 6 required questions
9. Limitations — dataset size, bias, scope
10. Conclusion
11. Ethics Statement — consent, bias, reproducibility
12. References — minimum 10 academic sources

---

## 9. Discussion Questions (must answer in report)

1. Does DSP preprocessing improve performance? Why or why not?
2. Which frequency bands contain the most discriminative information?
3. Does the filter remove any useful information?
4. How does preprocessing affect overfitting?
5. What is the computational complexity trade-off between Pipeline A and B?
6. Would DSP still be necessary if using deep learning?

---

## 10. Evaluation Criteria

| Criterion | Weight |
|---|---|
| DSP Design & Theoretical Justification | 25% |
| Before vs After DSP Comparison (A1 vs B1) | 20% |
| AI/ML Model Design (SVM) | 20% |
| Experimental Rigor | 15% |
| Critical Discussion | 10% |
| Report Quality | 5% |
| Innovation | 5% |
