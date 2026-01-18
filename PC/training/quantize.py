import tensorflow as tf
import numpy as np

# Load trained model
model = tf.keras.models.load_model('keyword_cnn.h5')

# Load dataset for calibration
X = np.load("X.npy")
y = np.load("y.npy")

# Add channel dimension if missing (3D -> 4D)
if X.ndim == 3:
    X = X[..., np.newaxis]  # shape: (samples, time, mfcc, 1)

# Create TFLite converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Provide representative dataset for full integer quantization
def representative_dataset():
    for i in range(len(X)):
        # Convert each sample to float32 and add batch dimension
        yield [X[i:i+1].astype(np.float32)]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

# Convert and save
tflite_model = converter.convert()
with open("kws_model_quant.tflite", "wb") as f:
    f.write(tflite_model)

print("Quantized model saved as kws_model_quant.tflite")
