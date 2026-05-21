import librosa
import numpy as np
import os

DATASET_PATH = r'C:\Users\paran\AudioShield\dataset_filtered'
TARGET_CLASSES = ["background", "glass_breaking", "knocking", "raining", "speaking"]
SR = 22050
N_MELS = 128
TARGET_WIDTH = 87

data = []
labels = []

print("Extracting features...")

for label_idx, class_name in enumerate(TARGET_CLASSES):
    folder_path = os.path.join(DATASET_PATH, class_name)
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(folder_path, filename)
            try:
                y, _ = librosa.load(file_path, sr=SR)
                y = librosa.util.normalize(y)
                mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=2048, hop_length=512)
                mel_db = librosa.power_to_db(mel, ref=np.max)
                if mel_db.shape[1] < TARGET_WIDTH:
                    mel_db = np.pad(mel_db, ((0,0), (0, TARGET_WIDTH - mel_db.shape[1])))
                else:
                    mel_db = mel_db[:, :TARGET_WIDTH]
                data.append(mel_db)
                labels.append(label_idx)
            except Exception as e:
                print(f"Skipping corrupt file {filename}: {e}")

np.save('X.npy', np.array(data))
np.save('y.npy', np.array(labels))
print(f"Extraction complete. {len(data)} samples saved.")