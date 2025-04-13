import os
import numpy as np
import tensorflow as tf
import librosa
import librosa.display
import sounddevice as sd
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import time  # Import time to generate unique filenames
from tensorflow.keras.preprocessing import image

# Paths
MODEL_PATH = "models/infant_cry_classifier.h5"
AUDIO_DIR = "recordings/"  # Folder to save all audio files
SPECTROGRAM_DIR = "test_spectrogram/"  # Folder to save spectrograms

# Ensure directories exist
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(SPECTROGRAM_DIR, exist_ok=True)

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Define Baby Cry Categories
CATEGORIES = ["belly_pain", "hungry", "cold_hot", "tired", "silence", "discomfort", "burping"]

# Confidence threshold for rejecting uncertain predictions
CONFIDENCE_THRESHOLD = 0.60  # Adjust as needed

# **📌 Step 1: Record Audio (Unique File)**
def record_audio(duration=10, samplerate=22050):
    """
    Records audio from the microphone and saves it with a unique filename.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")  # Generate timestamp
    audio_path = os.path.join(AUDIO_DIR, f"audio_{timestamp}.wav")

    print(f"🎤 Recording for {duration} seconds...")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    wav.write(audio_path, samplerate, audio_data)
    
    print(f"✅ Audio recorded and saved as '{audio_path}'")
    return audio_path  # Return the unique filename

# **📌 Step 2: Convert Audio to Spectrogram (Unique File)**
def audio_to_spectrogram(audio_path):
    """
    Converts an audio file into a spectrogram image with a unique filename.
    """
    timestamp = os.path.basename(audio_path).replace(".wav", "")  # Extract timestamp from audio filename
    spectrogram_path = os.path.join(SPECTROGRAM_DIR, f"spectrogram_{timestamp}.png")

    y, sr = librosa.load(audio_path, sr=22050)
    plt.figure(figsize=(5, 5))
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)

    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.savefig(spectrogram_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"📊 Spectrogram saved at '{spectrogram_path}'")
    return spectrogram_path  # Return the spectrogram filename

# **📌 Step 3: Predict from Spectrogram**
def predict_from_spectrogram(img_path):
    """
    Loads the spectrogram image, makes a prediction, and displays the result.
    """
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Model prediction
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)  # Get class with highest probability
    confidence_score = prediction[0][predicted_index]  # Get confidence score

    # Check if confidence is too low
    if confidence_score < CONFIDENCE_THRESHOLD:
        predicted_class = "Unknown Sound (Not a Baby Cry)"
    else:
        predicted_class = CATEGORIES[predicted_index]

    # Print detailed probabilities
    print(f"\n🍼 Prediction Probabilities: {prediction[0]}")
    print(f"🔎 Predicted Infant Cry Reason: {predicted_class} (Confidence: {confidence_score * 100:.2f}%)\n")

    # Show spectrogram with prediction
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicted: {predicted_class} ({confidence_score * 100:.2f}%)")
    plt.show()
CONFIDENCE_THRESHOLD = 0.6  # Reject predictions below this

def predict_from_spectrogram(img_path):
    """
    Loads the spectrogram image, makes a prediction, and displays the result.
    """
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Model prediction
    prediction = model.predict(img_array)
    
    # Print full prediction output
    print(f"\n🔍 Raw Model Output: {prediction}")

    # Find the index with the highest probability
    predicted_index = np.argmax(prediction)  
    confidence_score = prediction[0][predicted_index]  

    # **Reject low-confidence predictions**
    if confidence_score < CONFIDENCE_THRESHOLD:
        predicted_class = "Unknown Sound (Not a Baby Cry)"
    else:
        predicted_class = CATEGORIES[predicted_index]

    # Print detailed results
    print(f"\n🍼 Prediction Probabilities: {prediction[0]}")
    print(f"🔎 Predicted Infant Cry Reason: {predicted_class} (Confidence: {confidence_score * 100:.2f}%)\n")

    # Show spectrogram with prediction
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicted: {predicted_class} ({confidence_score * 100:.2f}%)")
    plt.show()

# **📌 Main Execution**
if __name__ == "__main__":
    audio_file = record_audio(duration=10)  # Record with unique filename
    spectrogram_file = audio_to_spectrogram(audio_file)  # Convert to unique spectrogram
    predict_from_spectrogram(spectrogram_file)  # Predict from the model


