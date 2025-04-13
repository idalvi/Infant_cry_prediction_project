import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Paths
MODEL_PATH = "models/infant_cry_classifier.h5"
TEST_IMAGE_PATH = "F:\Infant_Project\spectrograms"

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels (ensure this matches your training labels)
CATEGORIES = sorted(os.listdir("spectrograms/"))

# Function to make predictions
def predict_cry(img_path):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = CATEGORIES[np.argmax(prediction)]

    # Display image with prediction
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicted: {predicted_class}")
    plt.show()

# Test prediction
predict_cry(TEST_IMAGE_PATH)





