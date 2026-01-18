# create_dataset.py
import numpy as np
import os

# Labels for your keywords
labels = ['yes', 'no', 'background']

# Automatically get the project root relative to this script
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'features')

X = []
y = []

for idx, label in enumerate(labels):
    folder = os.path.join(BASE_DIR, label)
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    
    for file in os.listdir(folder):
        if file.endswith('.npy'):
            file_path = os.path.join(folder, file)
            X.append(np.load(file_path))
            y.append(idx)

X = np.array(X)
y = np.array(y)

# Save combined dataset in the training folder
OUTPUT_DIR = os.path.dirname(__file__)
np.save(os.path.join(OUTPUT_DIR, 'X.npy'), X)
np.save(os.path.join(OUTPUT_DIR, 'y.npy'), y)

print(f"X.npy and y.npy created in {OUTPUT_DIR}!")
print(f"Total samples: {len(y)}")
