import numpy as np
import sounddevice as sd
import librosa
import tflite_runtime.interpreter as tflite

SAMPLE_RATE = 16000
DURATION = 1.0

labels = ["YES", "NO", "BACKGROUND"]

interpreter = tflite.Interpreter(model_path="../models/kws_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def get_mfcc(audio):
    mfcc = librosa.feature.mfcc(audio, sr=SAMPLE_RATE, n_mfcc=13)
    mfcc = mfcc.T[:49]
    return mfcc[np.newaxis, ..., np.newaxis].astype(np.int8)

print("Listening... Press CTRL+C to stop")

while True:
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()

    mfcc = get_mfcc(audio.flatten())
    interpreter.set_tensor(input_details[0]['index'], mfcc)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    print("Detected:", labels[np.argmax(output)])
