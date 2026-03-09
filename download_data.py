"""
download_data.py
----------------
Download a small subset of LibriSpeech (test-clean) and prepare it
for the speaker identification pipeline.

What it does:
  1. Download LibriSpeech test-clean (~346 MB) from openslr.org
  2. Pick 5 speakers with the most recordings
  3. Take 10 utterances per speaker
  4. Convert .flac → .wav (16 kHz, mono, 3 s)
  5. Write data/index.csv

Run:
  python3 download_data.py
"""

import os
import csv
import tarfile
import urllib.request
import soundfile as sf
import numpy as np

# ── Config ─────────────────────────────────────────────────────────
URL = 'https://www.openslr.org/resources/12/test-clean.tar.gz'
ARCHIVE = 'test-clean.tar.gz'
EXTRACT_DIR = 'librispeech_tmp'
OUTPUT_DIR = 'data/raw'
INDEX_CSV = 'data/index.csv'
N_SPEAKERS = 5
N_FILES = 10
SR = 16000
TARGET_LEN = 48000   # 3 seconds


def download(url, dest):
    if os.path.exists(dest):
        print(f"  Already downloaded: {dest}")
        return
    print(f"  Downloading {url}")
    print("  (this is ~346 MB, please wait...)")

    def progress(count, block_size, total):
        pct = min(count * block_size / total * 100, 100)
        print(f"  {pct:.1f}%", end='\r')

    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print("\n  Download complete.")


def extract(archive, dest):
    if os.path.exists(dest):
        print(f"  Already extracted: {dest}")
        return
    print(f"  Extracting {archive} ...")
    with tarfile.open(archive, 'r:gz') as tar:
        tar.extractall(dest)
    print("  Extraction complete.")


def find_flac_files(root):
    """Walk directory and collect all .flac files grouped by speaker ID."""
    speakers = {}
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith('.flac'):
                # LibriSpeech structure: LibriSpeech/test-clean/{speaker_id}/...
                parts = dirpath.replace('\\', '/').split('/')
                # find the speaker ID (first numeric folder after test-clean)
                for i, part in enumerate(parts):
                    if part == 'test-clean' and i + 1 < len(parts):
                        spk = parts[i + 1]
                        break
                else:
                    continue
                speakers.setdefault(spk, []).append(os.path.join(dirpath, fn))
    return speakers


def to_wav(flac_path, wav_path, target_sr=SR, target_len=TARGET_LEN):
    """Convert .flac to .wav at target_sr, mono, fixed length."""
    data, sr = sf.read(flac_path)

    # stereo → mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # resample if needed
    if sr != target_sr:
        import librosa
        data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=target_sr)

    data = data.astype(np.float32)

    # normalize
    peak = np.max(np.abs(data))
    if peak > 0:
        data /= peak

    # pad or crop
    if len(data) < target_len:
        data = np.pad(data, (0, target_len - len(data)))
    else:
        data = data[:target_len]

    sf.write(wav_path, data, target_sr)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Download
    download(URL, ARCHIVE)

    # 2. Extract
    extract(ARCHIVE, EXTRACT_DIR)

    # 3. Find speakers
    libri_root = os.path.join(EXTRACT_DIR, 'LibriSpeech', 'test-clean')
    all_speakers = find_flac_files(libri_root)

    # Pick N_SPEAKERS with most files
    sorted_spk = sorted(all_speakers.items(), key=lambda x: len(x[1]), reverse=True)
    selected = sorted_spk[:N_SPEAKERS]

    print(f"\nSelected {N_SPEAKERS} speakers:")
    for spk_id, files in selected:
        print(f"  Speaker {spk_id}: {len(files)} files available")

    # 4. Convert and save
    rows = []
    speaker_names = [f"Speaker_{i+1:02d}" for i in range(N_SPEAKERS)]

    for label_id, (spk_id, files) in enumerate(selected):
        out_dir = os.path.join(OUTPUT_DIR, f"speaker_{label_id+1:02d}")
        os.makedirs(out_dir, exist_ok=True)
        name = speaker_names[label_id]

        for i, flac_path in enumerate(files[:N_FILES]):
            wav_name = f"{i+1:02d}.wav"
            wav_path = os.path.join(out_dir, wav_name)
            print(f"  Converting {flac_path} → {wav_path}")
            to_wav(flac_path, wav_path)
            rows.append({
                'filename': f"speaker_{label_id+1:02d}/{wav_name}",
                'speaker_id': label_id,
                'speaker_name': name,
            })

    # 5. Write index.csv
    with open(INDEX_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'speaker_id', 'speaker_name'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone! {len(rows)} files written.")
    print(f"index.csv → {INDEX_CSV}")
    print("\nNext steps:")
    print("  cd src && python3 feature_extraction.py")
    print("  python3 train.py")
    print("  cd .. && streamlit run app.py")


if __name__ == '__main__':
    main()
