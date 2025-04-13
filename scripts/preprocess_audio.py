import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Paths
DATASET_PATH = "F:\Infant_Project\dataset"  # Folder containing 7 category folders
SPECTROGRAM_PATH = "F:\Infant_Project\spectrograms"  # Output folder for spectrogram images

# Ensure output directories exist
os.makedirs(SPECTROGRAM_PATH, exist_ok=True)

# Function to convert an audio file to a spectrogram
def audio_to_spectrogram(audio_file, output_file):
    try:
        # Load audio
        y, sr = librosa.load(audio_file, sr=22050)  

        # Generate Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  

        # Save spectrogram as an image
        plt.figure(figsize=(2, 2))
        librosa.display.specshow(mel_spec_db, sr=sr, cmap='inferno', fmax=8000)
        plt.axis('off')  
        plt.savefig(output_file, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close()
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")

# Process each category folder
for category in os.listdir(DATASET_PATH):
    category_path = os.path.join(DATASET_PATH, category)

    if os.path.isdir(category_path):  # Check if it's a folder
        output_category_path = os.path.join(SPECTROGRAM_PATH, category)
        os.makedirs(output_category_path, exist_ok=True)

        print(f"Processing category: {category}")

        # Process each audio file in the category
        for file in tqdm(os.listdir(category_path)):
            if file.endswith(".wav"):  # Ensure it's an audio file
                input_file = os.path.join(category_path, file)
                output_file = os.path.join(output_category_path, file.replace(".wav", ".png"))

                # Convert to spectrogram
                audio_to_spectrogram(input_file, output_file)

print("✅ Spectrogram generation completed!")
