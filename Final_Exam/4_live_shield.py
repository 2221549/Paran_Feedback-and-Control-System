import sounddevice as sd
import numpy as np
import librosa
import pyautogui
import time
import tensorflow as tf

MODEL_PATH = r'C:\Users\paran\AudioShield\glass_shield.h5'
DEVICE_INDEX = 1 
SR = 22050
TARGET_WIDTH = 87
CLASSES = ["background", "glass_breaking", "knocking", "raining", "speaking"]

model = tf.keras.models.load_model(MODEL_PATH)
audio_buffer = []

def callback(indata, frames, time_info, status):
    global audio_buffer
    audio_buffer.extend(indata[:, 0])
    if len(audio_buffer) > SR * 2:
        audio_buffer = audio_buffer[-(SR * 2):]

stream = sd.InputStream(device=DEVICE_INDEX, channels=1, samplerate=SR, callback=callback)
stream.start()

print("🛡️ AudioShield Active. Monitoring...")

try:
    while True:
        if len(audio_buffer) < SR * 2:
            time.sleep(0.1); continue
            
        live_audio = librosa.util.normalize(np.array(audio_buffer))
        mel = librosa.feature.melspectrogram(y=live_audio, sr=SR, n_mels=128, n_fft=2048, hop_length=512)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        if mel_db.shape[1] < TARGET_WIDTH:
            mel_db = np.pad(mel_db, ((0,0), (0, TARGET_WIDTH - mel_db.shape[1])))
        else:
            mel_db = mel_db[:, :TARGET_WIDTH]
        
        input_data = np.expand_dims(np.expand_dims(mel_db, axis=0), axis=-1)
        pred = model.predict(input_data, verbose=0)[0]
        
        if np.argmax(pred) == 1 and pred[1] > 0.85:
            print(f"🚨 SOUND DETECTED! Confidence: {pred[1]:.2f}")
            pyautogui.press('playpause')
            time.sleep(5)
            pyautogui.press('playpause')
            audio_buffer.clear()
            
        time.sleep(0.2)
except KeyboardInterrupt:
    stream.stop()
    print("Shield deactivated.")