import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# ---- Modell laden ----
model_size = "small"  # Alternativen: TINY, BASE, SMALL, MEDIUM, LARGE
model = WhisperModel(model_size, device="cpu")

# ---- Audio-Parameter ----
SAMPLE_RATE = 16000
BLOCK_SIZE = 1024

print("Sprich jetzt...")

audio_queue = []

def callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.append(indata[:, 0].copy())  # Mono

with sd.InputStream(channels=1, samplerate=SAMPLE_RATE,
                    blocksize=BLOCK_SIZE, callback=callback):
    
    while True:
        if not audio_queue:
            continue
        
        # Block zusammenfügen
        audio_data = np.concatenate(audio_queue).astype(np.float32)
        audio_queue = []  # leeren
        
        # ---- Whisper: Sprache auf Deutsch fixieren ----
        segments, info = model.transcribe(
            audio_data,
            language="de",       
            task="transcribe",    
            beam_size=5,
            temperature=0.0       
        )

        for segment in segments:
            print(segment.text)
