import os

# Path to spectrograms
SPECTROGRAM_PATH = "F:/Infant_Project/spectrograms"

# Count images in each category
category_counts = {}
for category in os.listdir(SPECTROGRAM_PATH):
    category_path = os.path.join(SPECTROGRAM_PATH, category)
    if os.path.isdir(category_path):
        count = len([file for file in os.listdir(category_path) if file.endswith(".png")])
        category_counts[category] = count

# Print dataset distribution
print("\n📊 Dataset Distribution:")
for category, count in category_counts.items():
    print(f"{category}: {count} images")
