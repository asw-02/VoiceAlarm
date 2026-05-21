import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# Load Whisper model
model_size = "small"  # Options: tiny, base, small, medium, large
model = WhisperModel(model_size, device="cpu")

# Audio settings
SAMPLE_RATE = 16000
BLOCK_SIZE = 1024

print("Speak now...")

audio_queue = []

def callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.append(indata[:, 0].copy())  # Mono audio

with sd.InputStream(
    channels=1,
    samplerate=SAMPLE_RATE,
    blocksize=BLOCK_SIZE,
    callback=callback
):
    while True:
        if not audio_queue:
            continue

        # Combine audio blocks
        audio_data = np.concatenate(audio_queue).astype(np.float32)
        audio_queue = []

        # Transcribe speech (German)
        segments, info = model.transcribe(
            audio_data,
            language="de",
            task="transcribe",
            beam_size=5,
            temperature=0.0
        )

        for segment in segments:
            print(segment.text)
