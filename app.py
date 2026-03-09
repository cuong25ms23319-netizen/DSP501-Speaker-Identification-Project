"""
app.py — Streamlit Demo
------------------------
Smart Meeting Room: identify who is speaking.

UI Flow:
  Sidebar  — registered speakers list
  Main     — upload .wav → choose pipeline → identify → show results
"""

import os
import sys
import tempfile
import numpy as np
import joblib
import matplotlib.pyplot as plt
import librosa
import librosa.display
import streamlit as st

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocess import preprocess
from filter import design_fir, apply_filter
from feature_extraction import extract_mfcc

# ── Config ─────────────────────────────────────────────────────────
SR = 16000
TARGET_LEN = 48000  # 3 seconds
MODEL_PATHS = {
    'Pipeline A (Raw)':      'models/svm_pipeline_a.pkl',
    'Pipeline B (Filtered)': 'models/svm_pipeline_b.pkl',
}
SPEAKER_NAMES = {0: 'Speaker 1', 1: 'Speaker 2', 2: 'Speaker 3',
                 3: 'Speaker 4', 4: 'Speaker 5', 5: 'Speaker 6'}


@st.cache_resource
def load_models():
    models = {}
    for name, path in MODEL_PATHS.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models


@st.cache_data
def get_fir():
    return design_fir()


# ── Page setup ──────────────────────────────────────────────────────
st.set_page_config(page_title='Speaker Identification', layout='wide')
st.title('🎙️ Speaker Identification System')
st.caption('DSP501 — FPT University | Pipeline A (Raw) vs Pipeline B (FIR Filtered)')

# ── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header('Registered Speakers')
    for sid, name in SPEAKER_NAMES.items():
        st.write(f'👤 {name}  (ID: {sid})')

    st.divider()
    st.header('Pipeline')
    pipeline_choice = st.radio('Select pipeline',
                                list(MODEL_PATHS.keys()),
                                index=0)
    show_both = st.checkbox('Compare both pipelines side by side', value=True)

# ── Main ─────────────────────────────────────────────────────────────
uploaded = st.file_uploader('Upload a .wav file (16 kHz, mono, ~3 s)', type=['wav'])

if uploaded:
    # Save to temp file so librosa can read it
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    st.audio(uploaded)

    # ── Preprocessing ──────────────────────────────────────────────
    y_raw, sr = preprocess(tmp_path, sr=SR, target_len=TARGET_LEN)
    coeffs = get_fir()
    y_filt = apply_filter(y_raw, coeffs)

    # ── Waveform plot ───────────────────────────────────────────────
    st.subheader('Waveform')
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.5), sharey=True)
    time = np.linspace(0, len(y_raw) / SR, len(y_raw))
    axes[0].plot(time, y_raw, linewidth=0.4)
    axes[0].set_title('Raw')
    axes[0].set_xlabel('Time (s)')
    axes[1].plot(time, y_filt, linewidth=0.4, color='orange')
    axes[1].set_title('Filtered (300–3400 Hz)')
    axes[1].set_xlabel('Time (s)')
    st.pyplot(fig)
    plt.close(fig)

    # ── MFCC heatmap ────────────────────────────────────────────────
    st.subheader('MFCC Heatmap')
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 3))
    for ax, y, title in zip(axes2, [y_raw, y_filt], ['Raw', 'Filtered']):
        mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13,
                                     n_fft=512, hop_length=256)
        img = librosa.display.specshow(mfcc, sr=SR, hop_length=256,
                                       x_axis='time', ax=ax, cmap='coolwarm')
        ax.set_title(f'MFCC — {title}')
        fig2.colorbar(img, ax=ax)
    st.pyplot(fig2)
    plt.close(fig2)

    # ── Feature extraction ──────────────────────────────────────────
    feat_raw  = extract_mfcc(y_raw).reshape(1, -1)
    feat_filt = extract_mfcc(y_filt).reshape(1, -1)

    models = load_models()

    def predict_with_pipeline(pipeline_name, feat):
        if pipeline_name not in models:
            return None, None
        model = models[pipeline_name]
        pred = model.predict(feat)[0]
        proba = model.predict_proba(feat)[0]
        return pred, proba

    # ── Results ─────────────────────────────────────────────────────
    st.subheader('Identification Result')

    if show_both:
        col_a, col_b = st.columns(2)
        cols_pipelines = [
            (col_a, 'Pipeline A (Raw)',      feat_raw),
            (col_b, 'Pipeline B (Filtered)', feat_filt),
        ]
    else:
        feat = feat_raw if 'A' in pipeline_choice else feat_filt
        cols_pipelines = [(st.container(), pipeline_choice, feat)]

    for col, pname, feat in cols_pipelines:
        with col:
            st.markdown(f'**{pname}**')
            pred, proba = predict_with_pipeline(pname, feat)
            if pred is None:
                st.warning('Model not found. Run train.py first.')
            else:
                speaker_name = SPEAKER_NAMES.get(pred, f'Speaker {pred}')
                confidence = float(proba[pred]) * 100
                st.success(f'Predicted: **{speaker_name}**')
                st.metric('Confidence', f'{confidence:.1f}%')
                # Confidence bar for each speaker
                fig3, ax = plt.subplots(figsize=(4, 2.5))
                labels = [SPEAKER_NAMES.get(i, str(i)) for i in range(len(proba))]
                ax.barh(labels, proba * 100, color='steelblue')
                ax.set_xlabel('Probability (%)')
                ax.set_title('Confidence per Speaker')
                ax.set_xlim(0, 100)
                st.pyplot(fig3)
                plt.close(fig3)

    os.unlink(tmp_path)

else:
    st.info('Upload a .wav file to start identification.')
