# Speaker Identification System
**DSP501 — FPT University**

Closed-set speaker identification comparing two pipelines:
- **Pipeline A** — Raw MFCC → SVM
- **Pipeline B** — FIR Filtered MFCC (300–3400 Hz) → SVM

---

## Yêu cầu

- Python 3.11 hoặc 3.12
- pip

---

## Bước 1 — Clone và tạo môi trường

```bash
git clone <repo-url>
cd dsp

python3.11 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

---

## Bước 2 — Cài thư viện

```bash
pip install -r requirements.txt
```

---

## Bước 3 — Chuẩn bị data

### Option A: Data giả (test nhanh, không cần mic)
```bash
python generate_fake_data.py
```
Sinh 5 speakers × 10 files sine wave, đủ để chạy thử toàn bộ pipeline.

### Option B: Download LibriSpeech (~346 MB, giọng người thật)
```bash
python download_data.py
```

### Option C: Tự thu âm
- Thu âm 5 người, mỗi người 10 file (~3 giây/file), 16kHz mono
- Đặt vào `data/raw/speaker_01/`, `speaker_02/`, ...
- Cập nhật `data/index.csv` nếu cần

---

## Bước 4 — Extract features

```bash
python src/feature_extraction.py
```

Tạo ra:
- `features/features_mfcc_raw.npy`
- `features/features_mfcc_filt.npy`
- `features/labels.npy`

---

## Bước 5 — Train models

```bash
python src/train.py
```

Tạo ra:
- `models/svm_pipeline_a.pkl`
- `models/svm_pipeline_b.pkl`
- `results.json`

---

## Bước 6 — Chạy Streamlit demo

```bash
streamlit run app.py
```

Mở trình duyệt tại `http://localhost:8501`, upload file `.wav` bất kỳ từ `data/raw/` để test.

---

## Bước 7 — Chạy Notebooks (cho báo cáo)

Mở Jupyter:
```bash
jupyter notebook
```

Chạy theo thứ tự:

| Notebook | Nội dung |
|---|---|
| `notebooks/01_analysis.ipynb` | Waveform, FFT, STFT, PSD, SNR |
| `notebooks/02_features.ipynb` | MFCC extraction walkthrough |
| `notebooks/03_train.ipynb`    | Training, CV scores, t-test |
| `notebooks/04_evaluation.ipynb` | Confusion matrix, ROC, bar chart |

---

## Cấu trúc project

```
dsp/
├── data/
│   ├── raw/                    ← .wav files (bị gitignore)
│   └── index.csv               ← danh sách file + speaker
├── features/                   ← auto-generated (bị gitignore)
├── models/                     ← auto-generated (bị gitignore)
├── figures/                    ← auto-generated (bị gitignore)
├── notebooks/
│   ├── 01_analysis.ipynb
│   ├── 02_features.ipynb
│   ├── 03_train.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── preprocess.py           ← load, normalize, trim, pad
│   ├── filter.py               ← FIR bandpass filter
│   ├── analysis.py             ← visualization functions
│   ├── feature_extraction.py   ← MFCC extraction
│   ├── train.py                ← SVM training + GridSearchCV
│   └── evaluation.py          ← metrics, plots
├── app.py                      ← Streamlit demo
├── generate_fake_data.py       ← sinh data giả để test
├── download_data.py            ← download LibriSpeech
├── requirements.txt
└── .gitignore
```

---

## Lưu ý

- `data/raw/`, `features/`, `models/` bị gitignore — **không commit data lên git**
- Mỗi người pull về phải chạy lại từ Bước 3
- Dùng cùng Python version để tránh lỗi thư viện
