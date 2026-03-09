# Speaker Identification System
**DSP501 — FPT University**

Closed-set speaker identification comparing two pipelines:
- **Pipeline A** — Raw MFCC → SVM
- **Pipeline B** — FIR Filtered MFCC → SVM

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Dataset

Place your `.wav` files (16 kHz, mono, ~3 s) in `data/raw/`:

```
data/raw/
  speaker_01/01.wav
  speaker_01/02.wav
  ...
  speaker_05/10.wav
```

Edit `data/index.csv` to match your actual filenames and speaker names.

---

## Run (in order)

### 1. Extract features
```bash
cd src
python feature_extraction.py
```
Saves `features/features_mfcc_raw.npy`, `features_mfcc_filt.npy`, `labels.npy`.

### 2. Train models
```bash
cd src
python train.py
```
Saves `models/svm_pipeline_a.pkl`, `svm_pipeline_b.pkl`, and `results.json`.

### 3. Launch Streamlit demo
```bash
streamlit run app.py
```

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `notebooks/01_analysis.ipynb`    | Signal analysis: waveform, spectrum, STFT |
| `notebooks/02_features.ipynb`    | MFCC extraction walkthrough |
| `notebooks/03_train.ipynb`       | Training + GridSearch results |
| `notebooks/04_evaluation.ipynb`  | Metrics, confusion matrix, ROC, t-test |

---

## Project Structure

```
dsp/
├── data/
│   ├── raw/              ← your .wav files here
│   └── index.csv
├── features/             ← auto-generated .npy files
├── models/               ← trained .pkl models
├── figures/              ← saved plots
├── notebooks/
├── src/
│   ├── preprocess.py
│   ├── filter.py
│   ├── analysis.py
│   ├── feature_extraction.py
│   ├── train.py
│   └── evaluation.py
├── app.py
├── results.json
└── requirements.txt
```
