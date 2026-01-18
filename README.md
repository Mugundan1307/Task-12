# Task-12 RPI Keyword Spotting

### Create a Python virtual environment (or) Use Anaconda
```
python3 -m venv kws_env
source kws_env/bin/activate
```

### Install core Python packages
```
pip install --upgrade pip setuptools wheel
pip install numpy scipy scikit-learn soundfile
pip install tflite-runtime  # For running TFLite on RPi
pip install tensorflow==2.12  # For training on Mac/PC
```

### Folder Structure

rpi-keyword-spotting/
│
├─ datasets/        # Raw audio recordings (yes, no, bg)
├─ features/        # MFCC features
├─ training/        # Scripts and dataset arrays
├─ models/          # Saved Keras/TFLite models
├─ inference/       # Live inference scripts

### Dataset
Use record.py script to capture keywords. And organize folder structure:

datasets/
│
├─ yes/
├─ no/
└─ bg/
