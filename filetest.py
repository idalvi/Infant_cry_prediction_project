import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

# Set parameters
samplerate = 22050  # Same as your training data
duration = 5  # Record for 5 seconds

print("🎤 Speak now...")
audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='int16')
sd.wait()
wav.write("test_mic.wav", samplerate, audio_data)
print("✅ Recording saved as 'test_mic.wav'")

# Check if recorded file has non-zero data
if np.any(audio_data):
    print("✅ Microphone is working correctly!")
else:
    print("❌ No audio detected! Check your microphone settings.")
