import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf

# Must match training
SR = 22050
DURATION = 2.5
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256
TARGET_LEN = int(SR * DURATION)

MODEL_PATH = "rob_keyword_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded!")

def record_audio(duration=DURATION, sr=SR):
    print(f"Get ready to speak... recording for {duration} seconds!")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    audio = audio.flatten()
    print("Recording finished!")
    return audio

def preprocess(audio):
    if len(audio) < TARGET_LEN:
        audio = np.pad(audio, (0, TARGET_LEN - len(audio)))
    else:
        audio = audio[:TARGET_LEN]

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

CONFIDENCE_THRESHOLD = 0.97  # Only detect if model confidence is above this

def predict(audio):
    mel = preprocess(audio)
    prediction = model.predict(mel)[0]  # shape: [2] for 0/1 probabilities
    prob_keyword = prediction[1]        # assuming index 1 is 'rob'
    
    if prob_keyword >= CONFIDENCE_THRESHOLD:
        return 1, prob_keyword
    else:
        return 0, prob_keyword

if __name__ == "__main__":
    audio = record_audio()
    result, conf = predict(audio)

    print(f"Prediction: {result}  Confidence: {conf:.4f}")
    result, confidence = predict(audio)
    if result == 1:
        print(f"✅ Detected: Confidence: {confidence:.2f}")
    else:
        print(f"❌ Not detected: Confidence: {confidence:.2f}")