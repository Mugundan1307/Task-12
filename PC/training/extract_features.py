import os
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import  spectrogram, resample
from scipy.signal.windows import hamming

# Settings
DATASET_DIR = "../dataset"
OUTPUT_DIR = "../features"
SAMPLE_RATE = 16000
NUM_MFCC = 13
FRAME_SIZE = 0.025  # 25ms
FRAME_STRIDE = 0.01  # 10ms
EPS = 1e-10

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def compute_mfcc(signal, sr=SAMPLE_RATE, num_mfcc=NUM_MFCC):
    # Pre-emphasis
    pre_emphasis = 0.97
    emphasized = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    # Framing
    frame_length = int(FRAME_SIZE * sr)
    frame_step = int(FRAME_STRIDE * sr)
    signal_length = len(emphasized)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)) + 1

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized, z)

    indices = (
        np.tile(np.arange(0, frame_length), (num_frames, 1)) +
        np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    )
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    frames *= hamming(frame_length, sym=False)

    # FFT + Power Spectrum
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))

    # Filter banks
    nfilt = 26
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sr / 2) / 700))
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin = np.floor((NFFT + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = bin[m - 1]
        f_m = bin[m]
        f_m_plus = bin[m + 1]
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1] + EPS)
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m] + EPS)

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, EPS, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)

    # DCT
    from scipy.fftpack import dct
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :num_mfcc]

    return mfcc

# Loop through dataset
for label in os.listdir(DATASET_DIR):
    label_path = os.path.join(DATASET_DIR, label)
    if not os.path.isdir(label_path):
        continue

    output_label_dir = os.path.join(OUTPUT_DIR, label)
    if not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)

    for file in os.listdir(label_path):
        if not file.lower().endswith(".wav"):
            continue
        filepath = os.path.join(label_path, file)
        sr, signal = wav.read(filepath)

        # Resample if needed
        if sr != SAMPLE_RATE:
            num_samples = int(len(signal) * SAMPLE_RATE / sr)
            signal = resample(signal, num_samples)

        mfcc_feat = compute_mfcc(signal)
        np.save(os.path.join(output_label_dir, file.replace(".wav", ".npy")), mfcc_feat)

print("MFCC extraction done!")
