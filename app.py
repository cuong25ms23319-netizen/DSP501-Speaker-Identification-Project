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
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocess import normalize, trim_silence, pad_or_crop
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
    """Convert uploaded audio bytes to 16kHz mono numpy array."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        y, _ = librosa.load(tmp_path, sr=SR, mono=True)
    finally:
        os.unlink(tmp_path)
    return y


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
    st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/5/51/FPT_logo_2010.svg/1200px-FPT_logo_2010.svg.png', width=120)
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
        @st.cache_resource
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
            y = process_audio(test_audio)
            if len(y) > TARGET_LEN:
                y = y[:TARGET_LEN]
            elif len(y) < TARGET_LEN:
                y = np.pad(y, (0, TARGET_LEN - len(y)))

            y_proc = normalize(y)
            y_proc = trim_silence(y_proc)
            y_proc = pad_or_crop(y_proc, TARGET_LEN)

            fir_coeffs = get_fir()
            y_filt = apply_filter(y_proc, fir_coeffs)
            y_filt = pre_emphasize(y_filt)

            feat_raw = extract_basic_features(y_proc).reshape(1, -1)
            feat_filt = extract_mfcc(y_filt).reshape(1, -1)

            st.markdown('---')
            st.audio(audio_to_bytes(y_proc), format='audio/wav')

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

            st.markdown('### 🎯 Kết quả nhận diện')
            col_a, col_b = st.columns(2)

            for col, pname, feat in [
                (col_a, 'Pipeline A (Raw)', feat_raw),
                (col_b, 'Pipeline B (Filtered)', feat_filt),
            ]:
                with col:
                    model = models[pname]
                    pred = model.predict(feat)[0]
                    proba = model.predict_proba(feat)[0]
                    classes = model.classes_

                    pred_name = speaker_map.get(pred, f'Speaker {pred}')
                    confidence = float(proba[list(classes).index(pred)]) * 100

                    st.markdown(f'**{pname}**')

                    if confidence >= 70:
                        st.success(f'🎤 **{pred_name}** — {confidence:.1f}%')
                    elif confidence >= 50:
                        st.warning(f'🤔 **{pred_name}** — {confidence:.1f}%')
                    else:
                        st.error(f'❓ **{pred_name}** — {confidence:.1f}% (không chắc)')

                    fig, ax = plt.subplots(figsize=(5, max(2, len(classes) * 0.4)))
                    labels = [speaker_map.get(c, f'Speaker {c}') for c in classes]
                    colors = ['#2ecc71' if c == pred else '#95a5a6' for c in classes]
                    bars = ax.barh(labels, proba * 100, color=colors)
                    ax.set_xlabel('Confidence (%)')
                    ax.set_xlim(0, 100)
                    for bar, p in zip(bars, proba):
                        if p > 0.05:
                            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                                    f'{p * 100:.1f}%', va='center', fontsize=9)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

            st.markdown('---')
            pred_a = models['Pipeline A (Raw)'].predict(feat_raw)[0]
            pred_b = models['Pipeline B (Filtered)'].predict(feat_filt)[0]
            name_a = speaker_map.get(pred_a, f'Speaker {pred_a}')
            name_b = speaker_map.get(pred_b, f'Speaker {pred_b}')

            if pred_a == pred_b:
                st.success(f'✅ Cả 2 pipeline đều nhận diện: **{name_a}**')
            else:
                st.warning(f'⚠️ Pipeline A: **{name_a}** | Pipeline B: **{name_b}** — Kết quả khác nhau!')


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
                    if st.button('✅ Xác nhận xóa tất cả', key='confirm_all_yes'):
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

    st.divider()
    st.markdown('#### 📤 Chia sẻ data cho nhóm (Git)')
    st.code('''# Sau khi thu âm xong, push data lên GitHub:
git add data/
git commit -m "Add audio data for [tên bạn]"
git push

# Các bạn khác pull về — app tự nhận data mới:
git pull
streamlit run app.py''', language='bash')
