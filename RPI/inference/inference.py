import numpy as np
import sounddevice as sd
import tflite_runtime.interpreter as tflite
from scipy.io import wavfile
from python_speech_features import mfcc
import time

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "../models/kws_model_quant.tflite"
SAMPLE_RATE = 16000
DURATION = 1  # seconds
NUM_MFCC = 13
FRAME_LENGTH = 0.025
FRAME_STEP = 0.010
MAX_PAD_LEN = 199
KEYWORDS = ["yes", "no", "bg"]

# -----------------------------
# Load TFLite model
# -----------------------------
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_dtype = input_details[0]['dtype']
input_index = input_details[0]['index']
output_index = output_details[0]['index']

# -----------------------------
# MFCC extraction
# -----------------------------
def extract_features(audio):
    mfcc_features = mfcc(audio, samplerate=SAMPLE_RATE, numcep=NUM_MFCC,
                         winlen=FRAME_LENGTH, winstep=FRAME_STEP)
    if len(mfcc_features) < MAX_PAD_LEN:
        pad_width = MAX_PAD_LEN - len(mfcc_features)
        mfcc_features = np.pad(mfcc_features, ((0, pad_width), (0, 0)), mode='constant')
    else:
        mfcc_features = mfcc_features[:MAX_PAD_LEN, :]
    return mfcc_features  # shape: (199, 13)

# -----------------------------
# Prediction
# -----------------------------
def predict(interpreter, features):
    input_data = np.expand_dims(features, axis=0)  # batch dim
    if input_data.ndim == 3:  # add channel dim
        input_data = np.expand_dims(input_data, axis=-1)  # shape: (1, 199, 13, 1)

    # Quantize if model expects UINT8
    if input_dtype == np.uint8:
        scale, zero_point = input_details[0]['quantization']
        input_data = input_data / scale + zero_point
        input_data = input_data.astype(np.uint8)

    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_index)[0]

    # Dequantize if needed
    if output_details[0]['dtype'] == np.uint8:
        scale, zero_point = output_details[0]['quantization']
        output = scale * (output - zero_point)

    return output  # array of 3 scores

# -----------------------------
# Record audio
# -----------------------------
def record_audio():
    print("\nPress ENTER to record a sample...")
    input()
    print("Recording...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    audio = audio.flatten()
    return audio

# -----------------------------
# Live loop
# -----------------------------
print("Live keyword spotting started! Press Ctrl+C to exit.\n")

try:
    while True:
        audio = record_audio()
        features = extract_features(audio)
        output = predict(interpreter, features)
        predicted_index = np.argmax(output)
        confidence = output[predicted_index]

        print(f"Detected: {KEYWORDS[predicted_index]} | Confidence: {confidence:.2f}")
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nExiting live inference.")
