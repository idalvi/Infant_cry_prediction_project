# Infant_cry_prediction_project
# 👶 Infant Cry Classification using Deep Learning

This deep learning project classifies infant cries into 7 emotional or physical need categories by analyzing their sound patterns using spectrograms. It uses a CNN-LSTM-based model trained on labeled spectrograms to identify reasons behind an infant's cry.

## 🧠 Problem Statement

Infants cannot communicate their needs verbally. By analyzing the sound of their cries, we aim to automatically detect the possible reason—such as hunger, discomfort, or pain—thus helping caregivers respond appropriately and efficiently.


## 🔊 Cry Categories

The model is trained to classify cries into the following categories:

1. `belly_pain`
2. `hungry`
3. `cold_hot`
4. `tired`
5. `silence`
6. `discomfort`
7. `burping`

Each category currently contains **108 labeled cry audio samples**.

---

## 🏗️ Model Architecture

The model uses a combination of **Convolutional Neural Networks (CNN)** for spatial feature extraction and **LSTM (Long Short-Term Memory)** layers for sequential analysis.

### Model Summary:
- Input: 128x128x3 Spectrogram Images
- Conv2D → MaxPooling → Conv2D → MaxPooling → Conv2D → MaxPooling
- Flatten → RepeatVector → LSTM → LSTM
- Dense → Dropout → Output (Softmax)

Loss Function: `categorical_crossentropy`  
Optimizer: `adam`  
Metrics: `accuracy`

---

## 🧪 How to Use

### 1. Clone the repository

```bash

---

### 🔊 Cry Categories

The model is trained to classify cries into the following categories:

1. `belly_pain`
2. `hungry`
3. `cold_hot`
4. `tired`
5. `silence`
6. `discomfort`
7. `burping`

Each category currently contains **108 labeled cry audio samples**.

---

## 🏗️ Model Architecture

The model uses a combination of **Convolutional Neural Networks (CNN)** for spatial feature extraction and **LSTM (Long Short-Term Memory)** layers for sequential analysis.

### Model Summary:
- Input: 128x128x3 Spectrogram Images
- Conv2D → MaxPooling → Conv2D → MaxPooling → Conv2D → MaxPooling
- Flatten → RepeatVector → LSTM → LSTM
- Dense → Dropout → Output (Softmax)

Loss Function: `categorical_crossentropy`  
Optimizer: `adam`  
Metrics: `accuracy`

---

## 🧪 How to Use

### 1. Clone the repository

```bash
git clone https://github.com/ishadalvi05/infant-cry-classification.git
cd infant-cry-classification

cd infant-cry-classification

### 2. Install dependencies
pip install -r requirements.txt

### 3.Preprocess dataset (audio → spectrograms)
python preprocess_model.py

### 4.Train the model
python train_model.py

### 5.Evaluate model performance
python evaluate_model.py

### 6.Real-time prediction
python real_time_prediction.py


###📊 Dataset Info
Audio Format: .wav

Sampling Rate: 22050 Hz

Spectrogram Type: Mel-Spectrogram

Image Size: 128x128 (RGB)

Balanced: Yes (108 samples per category)

###✅ Current Accuracy
Train Accuracy: ~90%+

Validation Accuracy: ~85-88% (may vary slightly depending on system and batch size)

###🚀 Future Improvements
Add more non-cry (noise) audio for better filtering.

Use Transfer Learning (e.g., ResNet or MobileNet) for better performance.

Deploy model using Streamlit or Flask.

Add spectrogram augmentation (e.g., time masking, frequency masking).

Use tflite or ONNX for mobile deployment.

###⚙️ Requirements
Python 3.8+

TensorFlow

Keras

librosa

sounddevice

matplotlib

numpy

tqdm

scikit-learn

Install with: pip install -r requirements.txt

###👤 Author
Isha Dalvi
B.Tech IT Student | Data Analyst Enthusiast
GitHub: https://github.com/idalvi
LinkedIn: https://www.linkedin.com/in/isha-dalvi-880298249/

