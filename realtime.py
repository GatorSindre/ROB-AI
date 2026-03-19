import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import time

# Settings — match your training exactly
SR = 22050
DURATION = 2.5          # Sliding window size in seconds
CHECK_INTERVAL = 1    # How often we check
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256
MODEL_PATH = "rob_keyword_model.keras"
CONF_THRESHOLD = 0.95

TARGET_LEN = int(SR * DURATION)  # number of samples in 2.5s

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded!")

def on_detected(confidence):
    print(f"✅ Detected your word! Confidence: {confidence:.3f}")

    #TODO Code that happens after i talk to rob

def preprocess(audio):
    if len(audio) < TARGET_LEN:
        audio = np.pad(audio, (0, TARGET_LEN - len(audio)))
    else:
        audio = audio[-TARGET_LEN:]  # make sure it's the last 2.5s

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = mel_db[np.newaxis, ..., np.newaxis]
    return mel_db.astype(np.float32)

def predict(audio):
    x = preprocess(audio)
    pred = model.predict(x, verbose=0)[0]
    confidence = float(pred[1])  # class 1 = "rob"
    return confidence

def real_time_listener():
    print("🎤 Starting real-time detection. Press Ctrl+C to stop.")
    buffer = np.zeros(TARGET_LEN, dtype=np.float32)

    def audio_callback(indata, frames, time_info, status):
        nonlocal buffer
        buffer = np.roll(buffer, -frames)
        buffer[-frames:] = indata[:, 0]

    stream = sd.InputStream(
        samplerate=SR, channels=1, dtype="float32",
        callback=audio_callback, blocksize=int(CHECK_INTERVAL*SR)
    )

    with stream:
        try:
            while True:
                time.sleep(CHECK_INTERVAL)  # wait before next check
                confidence = predict(buffer)
                if confidence >= CONF_THRESHOLD:
                    on_detected(confidence)
        except KeyboardInterrupt:
            print("\nStopped.")

if __name__ == "__main__":
    real_time_listener()