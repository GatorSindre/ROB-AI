import os
import signal
import numpy as np
import librosa
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# -------- CONFIG --------
DATASET_PATH = "dataset"
CLASSES = ["0", "1"]   # 0 = not rob, 1 = rob
SR = 22050
DURATION = 2.5
TARGET_LEN = int(SR * DURATION)

CHECKPOINT_PATH = "rob_checkpoint.keras"
FINAL_MODEL_PATH = "rob_keyword_model.keras"

# -------- DATA LOADING --------
def extract_features(path):
    audio, _ = librosa.load(path, sr=SR, mono=True)

    # pad or trim to exactly 2.5 seconds
    if len(audio) < TARGET_LEN:
        audio = np.pad(audio, (0, TARGET_LEN - len(audio)))
    else:
        audio = audio[:TARGET_LEN]

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_mels=64,
        n_fft=1024,
        hop_length=256
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

def load_dataset():
    X, y = [], []

    for label, cls in enumerate(CLASSES):
        folder = os.path.join(DATASET_PATH, cls)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Missing folder: {folder}")

        for file in os.listdir(folder):
            if file.lower().endswith(".m4a") or file.lower().endswith(".wav"):
                path = os.path.join(folder, file)
                try:
                    features = extract_features(path)
                    X.append(features)
                    y.append(label)
                    print(f"Loaded: {path}")
                except Exception as e:
                    print(f"Skipped {path}: {e}")

    X = np.array(X, dtype=np.float32)[..., np.newaxis]
    y = to_categorical(np.array(y), num_classes=len(CLASSES))

    return train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

print("Loading dataset...")
X_train, X_test, y_train, y_test = load_dataset()
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# -------- MODEL --------
input_shape = X_train.shape[1:]

model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(len(CLASSES), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------- SAVE HELPERS --------
def save_all():
    print("\nSaving model...")
    model.save(FINAL_MODEL_PATH)
    model.save(CHECKPOINT_PATH)
    print(f"Saved {FINAL_MODEL_PATH}")
    print(f"Saved {CHECKPOINT_PATH}")

def handle_interrupt(sig, frame):
    print("\nCtrl+C received.")
    save_all()
    raise SystemExit

signal.signal(signal.SIGINT, handle_interrupt)

# -------- TRAIN LOOP --------
print("Training... press Ctrl+C to save and stop.")

epoch = 0
while True:
    epoch += 1
    print(f"\nEpoch round {epoch}")
    history = model.fit(
        X_train,
        y_train,
        epochs=1,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # save after every epoch round
    model.save(CHECKPOINT_PATH)
    print(f"Checkpoint saved after round {epoch}")