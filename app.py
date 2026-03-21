"""
DSP Speaker Identification — Streamlit App
-------------------------------------------
Thu âm, train model, và nhận diện giọng nói.
Chia sẻ qua GitHub cho cả nhóm.

Chạy:
  streamlit run app.py
"""

import os
import sys
import io
import json
import shutil
import subprocess
import tempfile
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import joblib
import noisereduce as nr
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocess import preprocess as preprocess_file, normalize, trim_silence, pad_or_crop
from filter import design_fir, apply_filter
from preemphasis import pre_emphasize
from feature_extraction import extract_mfcc, extract_basic_features

# ── Config ────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, 'data', 'raw')
INDEX_CSV = os.path.join(ROOT, 'data', 'index.csv')
SR = 16000
DURATION = 3
TARGET_LEN = SR * DURATION
REQUIRED_SAMPLES = 25

# ── Mẫu transcript chuẩn ─────────────────────────────────────────────
TRANSCRIPT = [
    "Xin chào, tôi là thành viên của nhóm bốn đề tài DSP Identify Speaker",
    "Hôm nay thời tiết thật đẹp, nắng vàng rực rỡ khắp nơi",
    "Trường đại học FPT nằm ở thành phố Hồ Chí Minh",
    "Xử lý tín hiệu số là môn học rất thú vị và bổ ích",
    "Mỗi người có một giọng nói riêng biệt không ai giống ai",
    "Tần số cơ bản của giọng nam thường thấp hơn giọng nữ",
    "Biến đổi Fourier giúp phân tích phổ tần số âm thanh",
    "Bộ lọc thông dải chỉ cho phép một dải tần số đi qua",
    "Học máy kết hợp xử lý tín hiệu tạo ra ứng dụng hay",
    "Cảm ơn bạn đã lắng nghe bài thuyết trình của nhóm tôi",
    "Nhận dạng người nói là bài toán quan trọng trong bảo mật",
    "Phòng họp thông minh có thể tự động ghi nhận ai đang nói",
    "Giọng nói mang nhiều thông tin về đặc điểm sinh học cá nhân",
    "Hệ số MFCC mô tả hình dạng đường thanh quản của người nói",
    "Thuật toán SVM tìm siêu phẳng phân tách tối ưu giữa các lớp",
    "Tiền xử lý tín hiệu giúp loại bỏ nhiễu và cải thiện chất lượng",
    "Cửa sổ Hamming giảm hiện tượng rò rỉ phổ khi tính FFT",
    "Tần số lấy mẫu mười sáu nghìn Hertz là chuẩn cho xử lý giọng nói",
    "Kiểm định thống kê giúp đánh giá kết quả một cách khách quan",
    "Dữ liệu huấn luyện cần đa dạng để mô hình tổng quát hóa tốt",
    "Việt Nam có năm mươi tư dân tộc với nhiều giọng nói khác nhau",
    "Trí tuệ nhân tạo đang thay đổi cách chúng ta sống và làm việc",
    "Ứng dụng nhận dạng giọng nói ngày càng phổ biến trên điện thoại",
    "Chất lượng microphone ảnh hưởng lớn đến độ chính xác nhận dạng",
    "Đây là câu cuối cùng trong bộ hai mươi lăm câu thu âm mẫu",
]


# ── Helpers ───────────────────────────────────────────────────────────

def load_index():
    if not os.path.exists(INDEX_CSV):
        return pd.DataFrame(columns=['filename', 'speaker_id', 'speaker_name'])
    return pd.read_csv(INDEX_CSV)


def save_index(df):
    os.makedirs(os.path.dirname(INDEX_CSV), exist_ok=True)
    df.to_csv(INDEX_CSV, index=False)


def get_speakers():
    df = load_index()
    if df.empty:
        return pd.DataFrame(columns=['speaker_id', 'speaker_name', 'folder', 'num_files'])
    speakers = df.groupby(['speaker_id', 'speaker_name']).agg(
        num_files=('filename', 'count'),
        folder=('filename', lambda x: x.iloc[0].split('/')[0])
    ).reset_index()
    return speakers.sort_values('speaker_id')


def sync_index_from_disk():
    """Scan data/raw/ for new audio files added by teammates via git pull."""
    if not os.path.exists(DATA_DIR):
        return 0

    df = load_index()
    existing_files = set(df['filename'].tolist()) if not df.empty else set()

    new_rows = []
    for folder_name in sorted(os.listdir(DATA_DIR)):
        folder_path = os.path.join(DATA_DIR, folder_name)
        if not os.path.isdir(folder_path) or not folder_name.startswith('speaker_'):
            continue

        meta_path = os.path.join(folder_path, '_meta.json')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            speaker_id = meta['speaker_id']
            speaker_name = meta['speaker_name']
        else:
            folder_rows = df[df['filename'].str.startswith(folder_name + '/')]
            if not folder_rows.empty:
                speaker_id = int(folder_rows.iloc[0]['speaker_id'])
                speaker_name = folder_rows.iloc[0]['speaker_name']
            else:
                continue

        for wav_name in sorted(os.listdir(folder_path)):
            if not wav_name.endswith('.wav'):
                continue
            rel_path = f"{folder_name}/{wav_name}"
            if rel_path not in existing_files:
                new_rows.append({
                    'filename': rel_path,
                    'speaker_id': speaker_id,
                    'speaker_name': speaker_name,
                })

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        save_index(df)
        return len(new_rows)
    return 0


def process_audio(audio_bytes):
    """Convert uploaded audio bytes to 16kHz mono numpy array.
    Handles WAV (from file upload) and WebM/OGG (from browser mic)."""
    # Try soundfile first (handles WAV natively)
    try:
        buf = io.BytesIO(audio_bytes)
        y, orig_sr = sf.read(buf, dtype='float32')
        if y.ndim > 1:
            y = y.mean(axis=1)
        if orig_sr != SR:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=SR)
        return y
    except Exception:
        pass

    # Fallback: temp file with correct extension for ffmpeg decode
    for ext in ['.webm', '.ogg', '.wav', '.mp3']:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            y, _ = librosa.load(tmp_path, sr=SR, mono=True)
            return y
        except Exception:
            continue
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    raise ValueError("Không thể đọc file audio")


def split_into_chunks(y, chunk_len=TARGET_LEN, min_len=SR):
    """Split audio into fixed-length chunks."""
    chunks = []
    for start in range(0, len(y), chunk_len):
        chunk = y[start:start + chunk_len]
        if len(chunk) < min_len:
            continue
        if len(chunk) < chunk_len:
            chunk = np.pad(chunk, (0, chunk_len - len(chunk)))
        chunks.append(chunk)
    return chunks


def save_audio_chunk(audio, speaker_folder, file_num):
    """Save a single audio chunk as wav."""
    rel_path = f"{speaker_folder}/{file_num:02d}.wav"
    full_path = os.path.join(DATA_DIR, rel_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    sf.write(full_path, audio, SR)
    return rel_path


def save_speaker_meta(folder, speaker_id, speaker_name):
    """Save metadata file in speaker folder for cross-team sync."""
    meta_path = os.path.join(DATA_DIR, folder, '_meta.json')
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, 'w') as f:
        json.dump({'speaker_id': speaker_id, 'speaker_name': speaker_name}, f)


def audio_to_bytes(y, sr=SR):
    """Convert numpy array to wav bytes for st.audio."""
    buf = io.BytesIO()
    sf.write(buf, y, sr, format='WAV')
    buf.seek(0)
    return buf.read()



# ── Auto-sync on load ────────────────────────────────────────────────
new_found = sync_index_from_disk()

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title='DSP Speaker Identification',
    layout='wide',
    page_icon='🎤',
    initial_sidebar_state='expanded',
)

st.markdown('''
<style>
    .block-container { padding-top: 2rem; }
    [data-testid="stSidebar"] { min-width: 280px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        border-radius: 8px 8px 0 0;
    }
</style>
''', unsafe_allow_html=True)

st.title('DSP Speaker Identification')
st.caption('DSP501 — FPT University | Nhóm 4 | MFCC + SVM Pipeline')

if new_found:
    st.toast(f'🔄 Phát hiện {new_found} file audio mới từ Git!', icon='🆕')

# ── Sidebar: speaker list ────────────────────────────────────────────
with st.sidebar:
    st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/a/a1/FPT_logo_2024.svg/1200px-FPT_logo_2024.svg.png', width=120)
    st.markdown('### Speakers')
    speakers = get_speakers()
    if speakers.empty:
        st.caption('Chưa có speaker nào.')
    else:
        for _, row in speakers.iterrows():
            n = int(row['num_files'])
            icon = '🟢' if n >= REQUIRED_SAMPLES else '🟡'
            st.markdown(f"{icon} **{row['speaker_name']}** — {n} files")

        st.divider()
        col_s1, col_s2 = st.columns(2)
        col_s1.metric('Speakers', len(speakers))
        col_s2.metric('Files', int(speakers['num_files'].sum()))

        ready_count = (speakers['num_files'] >= REQUIRED_SAMPLES).sum()
        if ready_count == len(speakers) and len(speakers) >= 2:
            st.success('Sẵn sàng Train')

    st.divider()
    if st.button('Sync data từ Git', use_container_width=True, icon='🔄'):
        n = sync_index_from_disk()
        if n:
            st.success(f'Tìm thấy {n} file mới!')
            st.rerun()
        else:
            st.info('Không có file mới.')

    st.divider()
    st.caption('streamlit run app.py')

# ── Tabs ──────────────────────────────────────────────────────────────
tab_test, tab_record, tab_upload, tab_manage, tab_train = st.tabs([
    'Test', 'Thu âm', 'Upload', 'Quản lý', 'Train'
])


# ═══════════════════════════════════════════════════════════════════════
# TAB: Test nhận diện speaker
# ═══════════════════════════════════════════════════════════════════════
with tab_test:
    st.subheader('Test nhận diện Speaker')

    model_a_path = os.path.join(ROOT, 'models', 'svm_pipeline_a.pkl')
    model_b_path = os.path.join(ROOT, 'models', 'svm_pipeline_b.pkl')

    if not (os.path.exists(model_a_path) and os.path.exists(model_b_path)):
        st.error('Chưa có model! Hãy vào tab **Train** để train trước.')
    else:
        def load_models():
            return {
                'Pipeline A (Raw)': joblib.load(model_a_path),
                'Pipeline B (Filtered)': joblib.load(model_b_path),
            }

        @st.cache_data
        def get_fir():
            return design_fir()

        models = load_models()
        df = load_index()
        speaker_map = dict(zip(df['speaker_id'].astype(int), df['speaker_name']))

        st.markdown('Nói vào mic hoặc upload file để xem model nhận diện bạn là ai!')
        test_mode = st.radio('Chọn cách test:', ['🎙️ Thu âm từ mic', '📁 Upload file'],
                              horizontal=True, key='test_mode')

        test_audio = None
        if test_mode == '🎙️ Thu âm từ mic':
            audio_input = st.audio_input('Nhấn để thu âm test 🎙️', key='test_mic')
            if audio_input:
                test_audio = audio_input.read()
        else:
            uploaded = st.file_uploader('Upload file audio',
                                         type=['wav', 'mp3', 'm4a', 'flac', 'ogg'],
                                         key='test_upload')
            if uploaded:
                test_audio = uploaded.read()

        if test_audio is not None:
            y_raw = process_audio(test_audio)

            # Nếu audio ngắn hơn 3s, lặp lại thay vì pad zero
            if 0 < len(y_raw) < TARGET_LEN:
                repeats = int(np.ceil(TARGET_LEN / len(y_raw)))
                y_raw = np.tile(y_raw, repeats)[:TARGET_LEN]

            # NR #1: giống recording tab (training data cũng bị NR ở bước này)
            y_raw = nr.reduce_noise(y=y_raw, sr=SR, stationary=True, prop_decrease=0.75)

            # Save temp WAV → preprocess() sẽ NR lần 2 (giống training pipeline)
            tmp_wav = os.path.join(ROOT, '_test_tmp.wav')
            sf.write(tmp_wav, y_raw, SR)
            try:
                y_proc, _ = preprocess_file(tmp_wav, sr=SR, target_len=TARGET_LEN)
            finally:
                if os.path.exists(tmp_wav):
                    os.unlink(tmp_wav)

            fir_coeffs = get_fir()
            y_filt = apply_filter(y_proc, fir_coeffs)
            y_filt = pre_emphasize(y_filt)

            # Pipeline A (Raw) uses basic time-domain features (6 dims)
            feat_raw = extract_basic_features(y_proc).reshape(1, -1)
            # Pipeline B (Filtered) uses MFCC features (26 dims)
            feat_filt = extract_mfcc(y_filt).reshape(1, -1)

            st.markdown('---')
            st.audio(audio_to_bytes(y_proc), format='audio/wav')

            # Debug info
            with st.expander('🔍 Debug info', expanded=False):
                st.text(f'Raw input: {len(y_raw)} samples ({len(y_raw)/SR:.2f}s)')
                st.text(f'After preprocess: {len(y_proc)} samples')
                st.text(f'Signal energy (RMS): {np.sqrt(np.mean(y_proc**2)):.4f}')
                st.text(f'MFCC[0:3]: {feat_filt[0,:3].round(2)}')

            with st.expander('📊 Phân tích tín hiệu', expanded=False):
                col_w1, col_w2 = st.columns(2)
                time_axis = np.linspace(0, len(y_proc) / SR, len(y_proc))

                with col_w1:
                    fig, ax = plt.subplots(figsize=(6, 2))
                    ax.plot(time_axis, y_proc, linewidth=0.5)
                    ax.set_title('Waveform (Raw)')
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Amplitude')
                    st.pyplot(fig)
                    plt.close(fig)

                with col_w2:
                    fig, ax = plt.subplots(figsize=(6, 2))
                    ax.plot(time_axis, y_filt, linewidth=0.5, color='orange')
                    ax.set_title('Waveform (Filtered 300-3400 Hz)')
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Amplitude')
                    st.pyplot(fig)
                    plt.close(fig)

                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    fig, ax = plt.subplots(figsize=(6, 3))
                    mfcc_raw = librosa.feature.mfcc(y=y_proc, sr=SR, n_mfcc=13,
                                                     n_fft=512, hop_length=256)
                    librosa.display.specshow(mfcc_raw, sr=SR, hop_length=256,
                                              x_axis='time', ax=ax, cmap='coolwarm')
                    ax.set_title('MFCC — Raw')
                    st.pyplot(fig)
                    plt.close(fig)

                with col_m2:
                    fig, ax = plt.subplots(figsize=(6, 3))
                    mfcc_filt = librosa.feature.mfcc(y=y_filt, sr=SR, n_mfcc=13,
                                                      n_fft=512, hop_length=256)
                    librosa.display.specshow(mfcc_filt, sr=SR, hop_length=256,
                                              x_axis='time', ax=ax, cmap='coolwarm')
                    ax.set_title('MFCC — Filtered')
                    st.pyplot(fig)
                    plt.close(fig)

            # ── Pipeline B: Main prediction (DSP Enhanced) ──────────
            model_b = models['Pipeline B (Filtered)']
            pred_b = model_b.predict(feat_filt)[0]
            proba_b = model_b.predict_proba(feat_filt)[0]
            classes_b = model_b.classes_
            name_b = speaker_map.get(pred_b, f'Speaker {pred_b}')
            conf_b = float(proba_b[list(classes_b).index(pred_b)]) * 100

            # ── Pipeline A: Baseline prediction ──────────────────
            model_a = models['Pipeline A (Raw)']
            pred_a = model_a.predict(feat_raw)[0]
            proba_a = model_a.predict_proba(feat_raw)[0]
            classes_a = model_a.classes_
            name_a = speaker_map.get(pred_a, f'Speaker {pred_a}')
            conf_a = float(proba_a[list(classes_a).index(pred_a)]) * 100

            # ── Hero result card ─────────────────────────────────
            if conf_b >= 70:
                border_color = '#2ecc71'
                emoji = '🎯'
                status_text = 'Độ tin cậy cao'
            elif conf_b >= 50:
                border_color = '#f39c12'
                emoji = '🤔'
                status_text = 'Độ tin cậy trung bình'
            else:
                border_color = '#e74c3c'
                emoji = '❓'
                status_text = 'Không chắc chắn'

            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                border-left: 6px solid {border_color};
                border-radius: 16px;
                padding: 30px;
                margin: 20px 0;
                box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            ">
                <div style="text-align: center;">
                    <span style="font-size: 48px;">{emoji}</span>
                    <h1 style="color: white; margin: 10px 0 5px 0; font-size: 2.2em;">
                        {name_b}
                    </h1>
                    <div style="
                        display: inline-block;
                        background: {border_color};
                        color: white;
                        padding: 6px 20px;
                        border-radius: 20px;
                        font-size: 1.1em;
                        font-weight: bold;
                    ">
                        {conf_b:.1f}% — {status_text}
                    </div>
                    <p style="color: #8899aa; margin-top: 12px; font-size: 0.9em;">
                        Pipeline B (FIR + Pre-emphasis + MFCC + SVM)
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Confidence bars for all speakers ─────────────────
            st.markdown('#### Phân bố xác suất theo Speaker')

            sorted_idx = np.argsort(proba_b)[::-1]
            for i, idx in enumerate(sorted_idx):
                spk_id = classes_b[idx]
                spk_name = speaker_map.get(spk_id, f'Speaker {spk_id}')
                prob = proba_b[idx] * 100
                is_pred = (spk_id == pred_b)

                if is_pred:
                    bar_color = border_color
                    label_style = f'color: {border_color}; font-weight: bold;'
                else:
                    bar_color = '#3a3a5c'
                    label_style = 'color: #8899aa;'

                col_name, col_bar, col_pct = st.columns([1.5, 6, 1])
                with col_name:
                    prefix = '▶ ' if is_pred else '  '
                    st.markdown(f'<span style="{label_style}">{prefix}{spk_name}</span>',
                                unsafe_allow_html=True)
                with col_bar:
                    st.progress(min(prob / 100, 1.0))
                with col_pct:
                    st.markdown(f'<span style="{label_style}">{prob:.1f}%</span>',
                                unsafe_allow_html=True)

            # ── Pipeline comparison (collapsible) ────────────────
            with st.expander('🔬 So sánh Pipeline A vs Pipeline B', expanded=False):
                col_cmp_a, col_cmp_b = st.columns(2)

                with col_cmp_a:
                    st.markdown(f"""
                    <div style="background: #2d2d44; border-radius: 12px; padding: 16px; text-align: center;">
                        <p style="color: #8899aa; margin: 0; font-size: 0.85em;">PIPELINE A — Baseline</p>
                        <p style="color: #e74c3c; font-size: 0.75em;">Không có DSP (chỉ RMS + ZCR)</p>
                        <h2 style="color: white; margin: 8px 0;">{name_a}</h2>
                        <span style="background: {'#2ecc71' if conf_a >= 70 else '#e74c3c'}; color: white;
                               padding: 4px 12px; border-radius: 12px; font-size: 0.9em;">
                            {conf_a:.1f}%
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

                    fig, ax = plt.subplots(figsize=(5, max(2, len(classes_a) * 0.35)))
                    labels_a = [speaker_map.get(c, f'Spk {c}') for c in classes_a]
                    colors_a = ['#e74c3c' if c == pred_a else '#3a3a5c' for c in classes_a]
                    ax.barh(labels_a, proba_a * 100, color=colors_a, height=0.6)
                    ax.set_xlim(0, 100)
                    ax.set_xlabel('Confidence (%)')
                    ax.set_facecolor('#1a1a2e')
                    fig.patch.set_facecolor('#1a1a2e')
                    ax.tick_params(colors='#8899aa')
                    ax.xaxis.label.set_color('#8899aa')
                    for spine in ax.spines.values():
                        spine.set_color('#3a3a5c')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                with col_cmp_b:
                    st.markdown(f"""
                    <div style="background: #1a3a2e; border-radius: 12px; padding: 16px; text-align: center;">
                        <p style="color: #8899aa; margin: 0; font-size: 0.85em;">PIPELINE B — DSP Enhanced</p>
                        <p style="color: #2ecc71; font-size: 0.75em;">FIR + Pre-emphasis + MFCC</p>
                        <h2 style="color: white; margin: 8px 0;">{name_b}</h2>
                        <span style="background: {'#2ecc71' if conf_b >= 70 else '#e74c3c'}; color: white;
                               padding: 4px 12px; border-radius: 12px; font-size: 0.9em;">
                            {conf_b:.1f}%
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

                    fig, ax = plt.subplots(figsize=(5, max(2, len(classes_b) * 0.35)))
                    labels_b = [speaker_map.get(c, f'Spk {c}') for c in classes_b]
                    colors_b = ['#2ecc71' if c == pred_b else '#3a3a5c' for c in classes_b]
                    ax.barh(labels_b, proba_b * 100, color=colors_b, height=0.6)
                    ax.set_xlim(0, 100)
                    ax.set_xlabel('Confidence (%)')
                    ax.set_facecolor('#1a1a2e')
                    fig.patch.set_facecolor('#1a1a2e')
                    ax.tick_params(colors='#8899aa')
                    ax.xaxis.label.set_color('#8899aa')
                    for spine in ax.spines.values():
                        spine.set_color('#3a3a5c')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                # Verdict
                if pred_a == pred_b:
                    st.success(f'✅ Cả 2 pipeline đều nhận diện: **{name_b}**')
                else:
                    st.info(f'💡 Pipeline A: **{name_a}** ({conf_a:.0f}%) vs Pipeline B: **{name_b}** ({conf_b:.0f}%) — Pipeline B chính xác hơn nhờ DSP')

            # ── Greeting message (no internet needed) ────────────
            st.markdown('---')
            if conf_b >= 70:
                st.markdown(f"""
                <div style="background: linear-gradient(90deg, #1a3a2e, #16213e); border-radius: 12px; padding: 20px; text-align: center;">
                    <span style="font-size: 32px;">👋</span>
                    <p style="color: white; font-size: 1.3em; margin: 8px 0;">
                        Xin chào <strong>{name_b}</strong>! Rất vui được gặp bạn.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            elif conf_b >= 50:
                st.markdown(f"""
                <div style="background: linear-gradient(90deg, #3a2a1e, #2e2e16); border-radius: 12px; padding: 20px; text-align: center;">
                    <span style="font-size: 32px;">🤔</span>
                    <p style="color: white; font-size: 1.3em; margin: 8px 0;">
                        Chào bạn! Mình đoán bạn là <strong>{name_b}</strong>, đúng không?
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: linear-gradient(90deg, #3a1a1a, #2e1616); border-radius: 12px; padding: 20px; text-align: center;">
                    <span style="font-size: 32px;">❓</span>
                    <p style="color: white; font-size: 1.3em; margin: 8px 0;">
                        Xin lỗi, mình không nhận ra bạn rõ ràng.
                    </p>
                </div>
                """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# TAB: Thu âm theo transcript chuẩn
# ═══════════════════════════════════════════════════════════════════════
with tab_record:
    st.subheader('Thu âm theo mẫu transcript')

    if 'rec_samples' not in st.session_state:
        st.session_state.rec_samples = []
    if 'rec_name' not in st.session_state:
        st.session_state.rec_name = ''
    if 'rec_done' not in st.session_state:
        st.session_state.rec_done = False

    speakers = get_speakers()

    mode = st.radio('Bạn là:', ['Thành viên mới', 'Thành viên đã thu âm (thu thêm)'],
                    horizontal=True, key='rec_mode')

    if mode == 'Thành viên mới':
        rec_name = st.text_input('Nhập tên của bạn:', placeholder='VD: Cuong',
                                  key='input_rec_name')
        if rec_name:
            st.session_state.rec_name = rec_name.strip()
        current_count = 0
    else:
        if speakers.empty:
            st.warning('Chưa có speaker nào. Chọn "Thành viên mới".')
            st.stop()
        selected_name = st.selectbox('Chọn tên:', speakers['speaker_name'].tolist(),
                                      key='rec_existing')
        st.session_state.rec_name = selected_name
        row = speakers[speakers['speaker_name'] == selected_name].iloc[0]
        current_count = int(row['num_files'])

    name = st.session_state.rec_name
    total_recorded = current_count + len(st.session_state.rec_samples)
    remaining = max(0, REQUIRED_SAMPLES - total_recorded)

    st.divider()
    progress_val = min(total_recorded / REQUIRED_SAMPLES, 1.0)
    st.progress(progress_val, text=f'Tiến độ: {total_recorded}/{REQUIRED_SAMPLES} samples')

    if total_recorded >= REQUIRED_SAMPLES and st.session_state.rec_samples:
        st.session_state.rec_done = True

    if remaining > 0 or not st.session_state.rec_done:
        next_idx = total_recorded % len(TRANSCRIPT)
        transcript_text = TRANSCRIPT[next_idx]

        st.markdown('---')
        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown(f'#### 📝 Câu {total_recorded + 1}: Hãy đọc câu sau')
            st.info(f'**"{transcript_text}"**', icon='🗣️')
            st.caption(f'⏱️ Thu âm {DURATION} giây — Đọc to, rõ ràng, tốc độ bình thường')

        with col_right:
            st.markdown('#### 🎙️ Thu âm')
            audio_data = st.audio_input(
                f'Nhấn để thu câu {total_recorded + 1}',
                key=f'mic_{total_recorded}'
            )

            if audio_data is not None:
                audio_bytes = audio_data.read()
                y = process_audio(audio_bytes)

                # Noise reduction trước khi save — xử lý mic ồn
                y = nr.reduce_noise(y=y, sr=SR, stationary=True, prop_decrease=0.75)

                if len(y) > TARGET_LEN:
                    y = y[:TARGET_LEN]
                elif len(y) < TARGET_LEN:
                    y = np.pad(y, (0, TARGET_LEN - len(y)))

                st.audio(audio_to_bytes(y), format='audio/wav')

                if st.button('✅ Chấp nhận & tiếp tục', type='primary', key=f'accept_{total_recorded}'):
                    st.session_state.rec_samples.append((y, next_idx))
                    st.rerun()

        if st.session_state.rec_samples:
            st.markdown('---')
            st.markdown(f'#### Đã thu trong phiên này: {len(st.session_state.rec_samples)} samples')
            for i, (sample_audio, tidx) in enumerate(st.session_state.rec_samples):
                with st.expander(f'Sample {current_count + i + 1}: "{TRANSCRIPT[tidx][:40]}..."'):
                    st.audio(audio_to_bytes(sample_audio), format='audio/wav')

    if st.session_state.rec_samples:
        st.markdown('---')

        if total_recorded >= REQUIRED_SAMPLES:
            st.success(f'🎉 Đã đủ {REQUIRED_SAMPLES} samples! Bạn có thể lưu.')
        else:
            st.warning(f'Cần thêm {remaining} samples nữa. Bạn vẫn có thể lưu trước.')

        if mode == 'Thành viên mới':
            final_name = st.text_input('Xác nhận tên của bạn:', value=name,
                                        key='final_name')
        else:
            final_name = name

        col_save, col_reset = st.columns(2)

        with col_save:
            if st.button(f'💾 Lưu {len(st.session_state.rec_samples)} samples cho "{final_name}"',
                         type='primary', use_container_width=True):
                if not final_name or not final_name.strip():
                    st.error('Vui lòng nhập tên!')
                    st.stop()

                df = load_index()

                if mode == 'Thành viên mới':
                    speaker_id = int(df['speaker_id'].max() + 1) if not df.empty else 0
                    folder = f"speaker_{speaker_id + 1:02d}"
                else:
                    r = speakers[speakers['speaker_name'] == final_name].iloc[0]
                    speaker_id = int(r['speaker_id'])
                    folder = r['folder']

                save_speaker_meta(folder, speaker_id, final_name.strip())

                existing = len(df[df['speaker_id'] == speaker_id])
                new_rows = []

                for i, (audio, _) in enumerate(st.session_state.rec_samples):
                    file_num = existing + i + 1
                    rel_path = save_audio_chunk(audio, folder, file_num)
                    new_rows.append({
                        'filename': rel_path,
                        'speaker_id': speaker_id,
                        'speaker_name': final_name.strip(),
                    })

                df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
                save_index(df)

                saved_count = len(st.session_state.rec_samples)
                st.session_state.rec_samples = []
                st.session_state.rec_done = False
                st.success(f'Đã lưu {saved_count} samples cho **{final_name}**!')
                st.rerun()

        with col_reset:
            if st.button('🗑️ Hủy tất cả recordings', use_container_width=True):
                st.session_state.rec_samples = []
                st.session_state.rec_done = False
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════
# TAB: Upload files
# ═══════════════════════════════════════════════════════════════════════
with tab_upload:
    st.subheader('Upload audio files')
    st.markdown('Hỗ trợ: **wav, mp3, m4a, flac, ogg**. Tự động convert sang 16kHz mono, '
                f'cắt thành đoạn {DURATION}s.')

    col1, col2 = st.columns([1, 2])

    with col1:
        speakers = get_speakers()
        up_mode = st.radio('Chọn:', ['Speaker mới', 'Speaker đã có'],
                           horizontal=True, key='upload_mode')

        if up_mode == 'Speaker mới':
            up_name = st.text_input('Tên speaker:', placeholder='VD: Cuong',
                                     key='upload_name')
        else:
            if speakers.empty:
                st.warning('Chưa có speaker nào.')
                st.stop()
            up_selected = st.selectbox('Chọn speaker:', speakers['speaker_name'].tolist(),
                                       key='upload_select')

    with col2:
        uploaded_files = st.file_uploader(
            'Chọn files audio',
            type=['wav', 'mp3', 'm4a', 'flac', 'ogg'],
            accept_multiple_files=True
        )

        if uploaded_files:
            st.write(f'Đã chọn **{len(uploaded_files)}** files')

            if st.button('💾 Import tất cả', type='primary', key='import_btn'):
                df = load_index()

                if up_mode == 'Speaker mới':
                    if not up_name or not up_name.strip():
                        st.error('Vui lòng nhập tên!')
                        st.stop()
                    name = up_name.strip()
                    speaker_id = int(df['speaker_id'].max() + 1) if not df.empty else 0
                    folder = f"speaker_{speaker_id + 1:02d}"
                else:
                    name = up_selected
                    row = speakers[speakers['speaker_name'] == name].iloc[0]
                    speaker_id = int(row['speaker_id'])
                    folder = row['folder']

                save_speaker_meta(folder, speaker_id, name)

                existing = len(df[df['speaker_id'] == speaker_id])
                new_rows = []
                total_chunks = 0
                progress = st.progress(0, text='Đang xử lý...')

                for idx, f in enumerate(uploaded_files):
                    try:
                        audio_bytes = f.read()
                        y = process_audio(audio_bytes)
                        chunks = split_into_chunks(y)

                        for chunk in chunks:
                            total_chunks += 1
                            file_num = existing + total_chunks
                            rel_path = save_audio_chunk(chunk, folder, file_num)
                            new_rows.append({
                                'filename': rel_path,
                                'speaker_id': speaker_id,
                                'speaker_name': name,
                            })
                    except Exception as e:
                        st.warning(f'Lỗi xử lý {f.name}: {e}')

                    progress.progress((idx + 1) / len(uploaded_files),
                                       text=f'Đang xử lý... {idx + 1}/{len(uploaded_files)}')

                df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
                save_index(df)
                progress.empty()
                st.success(f'Đã import **{total_chunks}** đoạn audio cho **{name}**!')
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════
# TAB: Quản lý speakers
# ═══════════════════════════════════════════════════════════════════════
with tab_manage:
    st.subheader('Quản lý Speakers & Data')

    speakers = get_speakers()

    if speakers.empty:
        st.info('Chưa có data. Hãy thu âm hoặc upload files ở các tab khác.')
    else:
        display_df = speakers[['speaker_id', 'speaker_name', 'folder', 'num_files']].copy()
        display_df.columns = ['ID', 'Tên', 'Thư mục', 'Số files']
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.markdown('#### Trạng thái thu âm')
        for _, row in speakers.iterrows():
            n = int(row['num_files'])
            if n >= REQUIRED_SAMPLES:
                st.success(f"✅ **{row['speaker_name']}**: {n} files — Đủ!")
            else:
                st.warning(f"⏳ **{row['speaker_name']}**: {n}/{REQUIRED_SAMPLES} files "
                           f"— Cần thêm {REQUIRED_SAMPLES - n}")

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('#### Xóa một speaker')
            del_speaker = st.selectbox('Chọn speaker cần xóa:',
                                        speakers['speaker_name'].tolist(),
                                        key='del_select')

            if st.button('🗑️ Xóa speaker này', type='secondary'):
                st.session_state['confirm_delete'] = del_speaker

            if st.session_state.get('confirm_delete') == del_speaker:
                st.warning(f'Bạn chắc chắn muốn xóa **{del_speaker}**?')
                c1, c2 = st.columns(2)
                with c1:
                    if st.button('✅ Xác nhận xóa', key='confirm_yes'):
                        df = load_index()
                        row = speakers[speakers['speaker_name'] == del_speaker].iloc[0]
                        folder_path = os.path.join(DATA_DIR, row['folder'])
                        if os.path.exists(folder_path):
                            shutil.rmtree(folder_path)
                        df = df[df['speaker_name'] != del_speaker]
                        save_index(df)
                        st.session_state.pop('confirm_delete', None)
                        st.success(f'Đã xóa {del_speaker}!')
                        st.rerun()
                with c2:
                    if st.button('❌ Hủy', key='confirm_no'):
                        st.session_state.pop('confirm_delete', None)
                        st.rerun()

        with col2:
            st.markdown('#### Xóa tất cả data')
            st.markdown('Xóa toàn bộ data (bao gồm data giả cũ)')

            if st.button('🗑️ Xóa tất cả', type='secondary'):
                st.session_state['confirm_delete_all'] = True

            if st.session_state.get('confirm_delete_all'):
                st.warning('Bạn chắc chắn muốn xóa **TẤT CẢ** data?')
                c1, c2 = st.columns(2)
                with c1:
                    if st.button('Xác nhận xóa tất cả', key='confirm_all_yes'):
                        if os.path.exists(DATA_DIR):
                            shutil.rmtree(DATA_DIR)
                            os.makedirs(DATA_DIR)
                        save_index(pd.DataFrame(columns=['filename', 'speaker_id', 'speaker_name']))
                        st.session_state.pop('confirm_delete_all', None)
                        st.success('Đã xóa tất cả data!')
                        st.rerun()
                with c2:
                    if st.button('❌ Hủy', key='confirm_all_no'):
                        st.session_state.pop('confirm_delete_all', None)
                        st.rerun()


# ═══════════════════════════════════════════════════════════════════════
# TAB: Train model
# ═══════════════════════════════════════════════════════════════════════
with tab_train:
    st.subheader('Train Model')

    speakers = get_speakers()
    total_files = int(speakers['num_files'].sum()) if not speakers.empty else 0

    col1, col2, col3 = st.columns(3)
    col1.metric('Số speakers', len(speakers))
    col2.metric('Tổng files', total_files)
    col3.metric('Files/speaker (TB)',
                f"{total_files / len(speakers):.0f}" if len(speakers) > 0 else '0')

    if len(speakers) < 2:
        st.error('Cần ít nhất **2 speakers** để train!')
    elif total_files < len(speakers) * 5:
        st.warning(f'Nên có ít nhất **{len(speakers) * 5} files** '
                   f'(5/speaker). Hiện có {total_files}.')

    if not speakers.empty:
        st.markdown('#### Data overview')
        fig, ax = plt.subplots(figsize=(8, 3))
        colors = ['#2ecc71' if n >= REQUIRED_SAMPLES else '#e74c3c'
                  for n in speakers['num_files']]
        ax.barh(speakers['speaker_name'], speakers['num_files'], color=colors)
        ax.axvline(x=REQUIRED_SAMPLES, color='gray', linestyle='--', alpha=0.5,
                   label=f'Minimum ({REQUIRED_SAMPLES})')
        ax.set_xlabel('Số files')
        ax.set_title('Số lượng audio files / speaker')
        ax.legend()
        for i, v in enumerate(speakers['num_files']):
            ax.text(v + 0.2, i, str(int(v)), va='center')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    if len(speakers) >= 2:
        st.markdown('#### Bắt đầu training')
        st.markdown('''
        Pipeline sẽ chạy:
        1. **Extract features** — MFCC (13 coefficients x mean + std = 26 features)
        2. **Train SVM** — GridSearchCV + 5-fold Stratified CV
        3. **Save models** — `models/svm_pipeline_a.pkl` & `svm_pipeline_b.pkl`
        ''')

        if st.button('🚀 Bắt đầu Train', type='primary', use_container_width=True):
            with st.status('Đang training...', expanded=True) as status:
                st.write('⏳ Bước 1/2: Extracting features...')
                r1 = subprocess.run(
                    [sys.executable, os.path.join(ROOT, 'src', 'feature_extraction.py')],
                    cwd=ROOT, capture_output=True, text=True
                )
                if r1.returncode != 0:
                    st.error(f'Feature extraction thất bại!\n```\n{r1.stderr}\n```')
                    status.update(label='Training thất bại!', state='error')
                    st.stop()
                st.write('✅ Features extracted!')
                st.code(r1.stdout)

                st.write('⏳ Bước 2/2: Training models...')
                r2 = subprocess.run(
                    [sys.executable, os.path.join(ROOT, 'src', 'train.py')],
                    cwd=ROOT, capture_output=True, text=True
                )
                if r2.returncode != 0:
                    st.error(f'Training thất bại!\n```\n{r2.stderr}\n```')
                    status.update(label='Training thất bại!', state='error')
                    st.stop()
                st.write('✅ Training complete!')
                st.code(r2.stdout)

                status.update(label='Training hoàn tất!', state='complete')

            st.balloons()
            st.success('Model đã được lưu! Chạy `streamlit run app.py` để test.')

            results_path = os.path.join(ROOT, 'results.json')
            if os.path.exists(results_path):
                with open(results_path) as f:
                    results = json.load(f)

                st.markdown('#### Kết quả Training')
                col1, col2 = st.columns(2)
                for col, (exp_name, exp) in zip([col1, col2], results['experiments'].items()):
                    with col:
                        st.markdown(f'**{exp_name}**')
                        st.metric('Accuracy', f"{exp['accuracy']['mean']:.2%}")
                        st.metric('F1 Score', f"{exp['f1_macro']['mean']:.2%}")
                        st.write(f"Best params: `{exp['best_params']}`")


