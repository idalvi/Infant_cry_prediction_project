import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
MODEL_PATH = "models/infant_cry_classifier.h5"
SPECTROGRAM_PATH = "F:/Infant_Project/spectrograms"  # Change as needed

# Image properties
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Data preprocessing (same as training)
datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)

val_generator = datagen.flow_from_directory(
    SPECTROGRAM_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# Evaluate model on validation data
loss, accuracy = model.evaluate(val_generator)

print(f"✅ Validation Accuracy: {accuracy * 100:.2f}%")
print(f"✅ Validation Loss: {loss:.4f}")
