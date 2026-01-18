import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

FEATURE_DIR = "./features"
LABELS = ["yes", "no", "background"]

# Load data
X, y = [], []
for idx, label in enumerate(LABELS):
    folder = os.path.join(FEATURE_DIR, label)
    for file in os.listdir(folder):
        if file.endswith(".npy"):
            data = np.load(os.path.join(folder, file))
            X.append(data)
            y.append(idx)

X = np.array(X)
y = np.array(y)

# Ensure shape is (samples, height, width, channels)
X = X[..., np.newaxis]
y = to_categorical(y, num_classes=len(LABELS))

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build simple CNN
model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=X.shape[1:]),
    MaxPooling2D((2,2)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(LABELS), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=8)

# Save
model.save("keyword_cnn.h5")
print("Training done and model saved as keyword_cnn.h5")
