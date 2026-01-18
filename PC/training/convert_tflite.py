import tensorflow as tf

# Load trained Keras model
model = tf.keras.models.load_model("keyword_cnn.h5")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # quantization
tflite_model = converter.convert()

# Save TFLite model
with open("keyword_cnn.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved as keyword_cnn.tflite")
