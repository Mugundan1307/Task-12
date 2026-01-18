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

### Dataset recording in RPI
Use record.py script to capture keywords. And organize folder structure as yes, no and background

### Extract MFCC features in PC
```
cd ~/rpi-keyword-spotting/training
python extract_features.py
```
Output: MFCC features saved in features/

### Create dataset arrays in PC
```
python create.py
```
Output: X.npy, y.npy

### Train the Model in PC
```
python train.py
```
Output: Produces keyword_cnn.h5 or kws_model.h5

### Convert the .h5 to .tflite by using convert_tflite.py
```
python convert_tflite.py
```
Output: keyword_cnn.tflite

### Then quantize it using quantize.py
```
python quantize.py
```
Output: kws_model_quant.tflite
Move the .tflite file to the RPi models/ folder

### Live inference on RPI
```
cd ~/rpi-keyword-spotting/inference
python3 inference.py
```
