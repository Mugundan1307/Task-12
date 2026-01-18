import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("../models/kws_model.h5")
X = np.load("X.npy")

def rep_data():
    for i in range(100):
        yield [X[i:i+1].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_data
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open("../models/kws_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Quantized model saved")
