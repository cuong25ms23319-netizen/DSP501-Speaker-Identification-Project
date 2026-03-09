"""
generate_fake_data.py
---------------------
Generate synthetic audio files to test the pipeline without real data.

Each "speaker" is simulated by a unique combination of sine wave frequencies
(mimicking different vocal tract characteristics).

Output: data/raw/speaker_0X/  with 10 .wav files each
"""

import os
import csv
import numpy as np
import soundfile as sf

SR = 16000
DURATION = 3        # seconds
N_SPEAKERS = 5
N_FILES = 10

# Each speaker = unique set of dominant frequencies (Hz)
# Simulates different vocal characteristics
SPEAKER_PROFILES = [
    {'name': 'Alice',  'freqs': [150, 400,  900, 2200]},
    {'name': 'Bob',    'freqs': [120, 350,  800, 1800]},
    {'name': 'Carol',  'freqs': [200, 500, 1100, 2600]},
    {'name': 'David',  'freqs': [110, 300,  700, 1600]},
    {'name': 'Eve',    'freqs': [180, 450, 1000, 2400]},
]


def generate_audio(freqs, sr=SR, duration=DURATION, noise_level=0.05, seed=0):
    """
    Generate a signal as sum of sine waves + small random noise.
    seed ensures each file from the same speaker is slightly different.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    signal = np.zeros_like(t)
    for freq in freqs:
        # Slightly vary frequency per file to make files unique
        f = freq * (1 + rng.uniform(-0.05, 0.05))
        signal += np.sin(2 * np.pi * f * t)

    # Add a little noise
    signal += rng.normal(0, noise_level, len(signal))

    # Normalize to [-1, 1]
    signal /= np.max(np.abs(signal))
    return signal.astype(np.float32)


def main():
    os.makedirs('data/raw', exist_ok=True)
    rows = []

    for spk_idx, profile in enumerate(SPEAKER_PROFILES):
        spk_dir = os.path.join('data', 'raw', f'speaker_{spk_idx+1:02d}')
        os.makedirs(spk_dir, exist_ok=True)

        for i in range(N_FILES):
            audio = generate_audio(profile['freqs'], seed=spk_idx * 100 + i)
            filename = f'{i+1:02d}.wav'
            path = os.path.join(spk_dir, filename)
            sf.write(path, audio, SR)
            rows.append({
                'filename': f'speaker_{spk_idx+1:02d}/{filename}',
                'speaker_id': spk_idx,
                'speaker_name': profile['name'],
            })

        print(f"  Speaker {spk_idx+1} ({profile['name']}): {N_FILES} files created")

    # Write index.csv
    with open('data/index.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'speaker_id', 'speaker_name'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone! {len(rows)} files in data/raw/")
    print("index.csv updated.")
    print("\nNext:")
    print("  cd src && python3 feature_extraction.py")
    print("  python3 train.py")
    print("  cd .. && streamlit run app.py")


if __name__ == '__main__':
    main()
