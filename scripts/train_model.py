import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Paths
SPECTROGRAM_PATH = "F:\Infant_Project\spectrograms"  # Path to processed spectrogram images
MODEL_PATH = "models/infant_cry_classifier.h5"

# Ensure model directory exists
os.makedirs("models", exist_ok=True)

# Image properties
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20  # Adjust based on dataset size

# Define class labels (folder names must match these)
CATEGORIES = ["belly_pain", "hungry", "cold_hot", "tired", "silence", "discomfort", "burping"]

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2  # 80-20 Train-Test Split
)

# Load training and validation datasets
train_generator = datagen.flow_from_directory(
    SPECTROGRAM_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    SPECTROGRAM_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# **📌 CNN + LSTM Model**
def build_model(input_shape=(128, 128, 3), num_classes=len(CATEGORIES)):
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.RepeatVector(10),  # LSTM expects a sequence, so we repeat the feature vector
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(64),

        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')  # 7 classes
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Build model
model = build_model()

# Model Summary
model.summary()

# Train Model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# Save trained model
model.save(MODEL_PATH)
print(f"✅ Model saved as {MODEL_PATH}")

# Plot Training Results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()


