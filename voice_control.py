import os
import sys
import json
import re
import joblib
import numpy as np
import pyaudio
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from collections import deque

# STT Import
from vosk import Model, KaldiRecognizer


# ============================================================
# 1. CONFIGURATION & PATHS
# ============================================================

# Use raw strings (r"...") or forward slashes for paths to avoid Windows errors
BASE_DIR = "voice_control"

# NLU
NLU_MODEL_PATH = os.path.join(BASE_DIR, "nlu", "nlu_model.pkl")

# STT (Vosk)
STT_MODEL_PATH = os.path.join(BASE_DIR, "speech_to_text", "vosk-model-de-0.21")

# Wake Word
WAKE_MODEL_PATH = os.path.join(BASE_DIR, "wake_word_detection", "wake_word_model.pt")
WAKE_STATS_PATH = os.path.join(BASE_DIR, "wake_word_detection", "dataset_stats.pt")

# Audio Settings
SAMPLE_RATE = 16000
CHUNK_SIZE = 4000       # Buffer size for reading from microphone
BUFFER_DURATION = 1.1   # Seconds of history to keep for Wake Word
WAKE_CONFIDENCE = 0.85  # Probability threshold to activate

# ============================================================
# 2. NLU CLASS (Inference Only)
# ============================================================

class AlarmNLU:
    def __init__(self, model_path):
        self.model_path = model_path
        self.classifier = None
        self.CONFIDENCE_THRESHOLD = 0.55
        
        # --- Time & Number Logic ---
        self.word_to_num = {
            "eins": 1, "ein": 1, "eine": 1, "zwei": 2, "drei": 3, "vier": 4,
            "fünf": 5, "sechs": 6, "sieben": 7, "acht": 8, "neun": 9, "zehn": 10,
            "elf": 11, "zwölf": 12, "zwanzig": 20, "dreißig": 30, "vierzig": 40, "fünfzig": 50
        }

        self.time_patterns = [
            (r"halb\s+(\d{1,2})", lambda m: ((int(m.group(1))-1)%24, 30)),
            (r"viertel vor\s+(\d{1,2})", lambda m: ((int(m.group(1))-1)%24, 45)),
            (r"viertel nach\s+(\d{1,2})", lambda m: (int(m.group(1)), 15)),
            (r"(\d{1,2})[:\.](\d{2})", lambda m: (int(m.group(1)), int(m.group(2)))),
            (r"(\d{1,2}) uhr\s*(\d{1,2})?", lambda m: (int(m.group(1)), int(m.group(2) or 0))),
            (r"(\d{1,2}) vor (\d{1,2})", lambda m: ((int(m.group(2))-1)%24, 60-int(m.group(1)))),
            (r"(\d{1,2}) nach (\d{1,2})", lambda m: (int(m.group(2)), int(m.group(1))))
        ]

        self.weekday_map = {
            "montag": 0, "dienstag": 1, "mittwoch": 2, "donnerstag": 3,
            "freitag": 4, "samstag": 5, "sonntag": 6,
            "heute": "PLUS_0", "morgen": "PLUS_1", "übermorgen": "PLUS_2"
        }

        self.daytime_rangers =  {
            "früh": (5, 10),
            "morgens": (6, 10),
            "vormittags": (9, 12),
            "mittags": (12, 13),
            "nachmittags": (13, 16),
            "abends": (16, 22),
            "abend": (16, 22),
            "nachts": (22, 5),
            "nacht": (22, 5)
        }

        self._load_model()

    def _load_model(self):
        """Loads the pre-trained sklearn pipeline."""
        if not os.path.exists(self.model_path):
            print(f"[NLU Error] Model file not found: {self.model_path}")
            sys.exit(1)
        
        print(f"[NLU] Loading model from {self.model_path}...")
        try:
            self.classifier = joblib.load(self.model_path)
        except Exception as e:
            print(f"[NLU Error] Failed to load model: {e}")
            sys.exit(1)

    def _preprocess(self, text):
        """Clean text exactly like during training."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        tokens = text.split()
        return " ".join(tokens)

    def _text_to_int(self, text):
        text = text.lower().strip()
        if text.isdigit():
            return int(text)
        return self.word_to_num.get(text, None)
    
    def _extract_time(self, text):
        """
        Detects hour and minute from the input text using German time patterns.
        Returns tuple (hour, minute) or (None, None).
        """
        text_proc = " ".join(str(self._text_to_int(w) or w) for w in text.lower().split())
        for pattern, func in self.time_patterns:
            m = re.search(pattern, text_proc)
            if m:
                return func(m)
        return None, None

    def _extract_weekday(self, text):
        """
        Detects weekday from text based on WEEKDAY_MAP.
        Returns weekday number (0-6) or special PLUS_x string for relative days.
        """
        text = text.lower()
        for word, val in self.weekday_map.items():
            if word in text:
                return val
        return None

    def _extract_daytime(self, text):
        """
        Detects German daytime words (e.g., 'morgens', 'abends') from text.
        Returns the matched key or None.
        """
        text = text.lower()
        for word, _ in self.daytime_rangers.items():
            if word in text:
                return word
        return None

    def _apply_daytime(self, hour, minute, daytime):
        """
        Adjusts the extracted hour based on the detected daytime range.
        Returns adjusted hour (0-23) or None.
        """
        if hour is None or daytime is None:
            return hour, minute
        
        start, end = self.daytime_rangers[daytime]
        
        if start > end:
            if not (hour >= start or hour < end):
                hour += 0
        else:
            if not (start <= hour < end):
                hour += 12

        hour = hour % 24
        return hour, minute

    def parse(self, text):
        """
        Main parsing function:
        - Preprocesses text
        - Predicts intent with classifier
        - Extracts slots: hour, minute, weekday
        - Applies daytime adjustment
        Returns dictionary: {text, intent, confidence, slots}.
        """
        processed_input = self._preprocess(text)
        probas = self.classifier.predict_proba([processed_input])[0]
        best_idx = np.argmax(probas)
        max_proba = probas[best_idx]
        predicted_intent = self.classifier.classes_[best_idx]

        final_intent = predicted_intent
        if max_proba < self.CONFIDENCE_THRESHOLD or predicted_intent == "no_intent":
            final_intent = "unknown"

        hour, minute = self._extract_time(text)
        weekday = self._extract_weekday(text)
        daytime = self._extract_daytime(text)

        hour, minute = self._apply_daytime(hour, minute, daytime)

        return {
            "text": text,
            "intent": final_intent,
            "confidence": float(max_proba),
            "slots": {"hour": hour, "minute": minute, "weekday": weekday}
        }
# ============================================================
# 3. MAIN ASSISTANT CLASS
# ============================================================

class VoiceAssistant:
    def __init__(self):
        self.is_running = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[System] Running on: {self.device}")

        # --- 1. Load NLU ---
        self.nlu = AlarmNLU(NLU_MODEL_PATH)

        # --- 2. Load STT (Vosk) ---
        if not os.path.exists(STT_MODEL_PATH):
            print(f"[STT Error] Vosk model not found at {STT_MODEL_PATH}")
            sys.exit(1)
        print(f"[STT] Loading Vosk model...")
        self.stt_model = Model(STT_MODEL_PATH)
        # Limit vocabulary for better accuracy on commands
        grammar = ["wecker", "alarm", "stellen", "löschen", "wetter", "uhrzeit", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun", "zehn", "zwanzig", "dreißig", "halb", "viertel", "nach", "vor", "uhr", "bitte", "morgen", "heute"]
        self.recognizer = KaldiRecognizer(self.stt_model, SAMPLE_RATE, json.dumps(grammar))

        # --- 3. Load Wake Word Model ---
        print(f"[Wake Word] Loading model from {WAKE_MODEL_PATH}...")
        try:
            self.wake_model = torch.jit.load(WAKE_MODEL_PATH, map_location=self.device)
            self.wake_model.eval()
        except Exception as e:
            print(f"[Wake Word Error] Failed to load model: {e}")
            sys.exit(1)

        # Load Normalization Stats
        if os.path.exists(WAKE_STATS_PATH):
            print(f"[Wake Word] Loading stats from {WAKE_STATS_PATH}")
            stats = torch.load(WAKE_STATS_PATH, map_location=self.device)
            self.norm_mean = stats["mean"].to(self.device)
            self.norm_std = stats["std"].to(self.device)
        else:
            print("[Wake Word Warning] No stats found. Using default norm.")
            self.norm_mean = torch.tensor(0.0).to(self.device)
            self.norm_std = torch.tensor(1.0).to(self.device)

        # Audio Transforms (Must match training!)
        self.mel_transform = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=400,
            hop_length=160, 
            n_mels=64
        ).to(self.device)

        # Rolling Buffer: Stores last ~1 second of audio
        self.buffer_len = int(SAMPLE_RATE * BUFFER_DURATION)
        self.audio_buffer = deque(maxlen=self.buffer_len)

    def _preprocess_audio_chunk(self, audio_bytes):
        """
        Takes raw audio bytes, updates rolling buffer, returns model-ready tensor.
        Output Shape: [1, 1, 64, 44]
        """
        # 1. Convert bytes to float
        audio_ints = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_floats = audio_ints.astype(np.float32) / 32768.0
        
        # 2. Add to rolling buffer
        self.audio_buffer.extend(audio_floats)
        
        # 3. Only process if buffer is full
        if len(self.audio_buffer) < self.buffer_len:
            return None

        # 4. Create Tensor
        waveform = torch.tensor(list(self.audio_buffer), dtype=torch.float32).to(self.device)
        
        # 5. Compute Mel Spectrogram [64, Time]
        mel_spec = self.mel_transform(waveform)
        
        mel_spec = torchaudio.functional.amplitude_to_DB(
            mel_spec, 
            multiplier=10.0, 
            amin=1e-10, 
            db_multiplier=0.0, 
            top_db=80.0 # Standard fallback if not specified in training explicitly, but function handles it
        )

        # 7. Resize/Interpolate to expected frame count (110 frames for 1.1s)
        # Input shape currently: [64, Time]
        mel_spec = mel_spec.unsqueeze(0).unsqueeze(0) # [1, 1, 64, Time]
        
        # Target: 64 Mels, 110 Frames (Calculated from 1.1s * 16000 / 160 hop)
        mel_spec = F.interpolate(mel_spec, size=(64, 110), mode='bilinear', align_corners=False)
        mel_spec = mel_spec.squeeze(0).squeeze(0) # [64, 110]

        # 8. Normalize using loaded stats
        mel_spec = (mel_spec - self.norm_mean) / (self.norm_std + 1e-6)

        # 9. Add Batch/Channel dims -> [1, 1, 64, 110]
        return mel_spec.unsqueeze(0).unsqueeze(0)

    def _check_wake_word(self, audio_data):
        input_tensor = self._preprocess_audio_chunk(audio_data)
        
        if input_tensor is None:
            return False

        with torch.no_grad():
            output = self.wake_model(input_tensor)
            probs = torch.softmax(output, dim=1)
            
            # Based on your train.py: Index 0 = "wake_word"
            wake_probability = probs[0][0].item()
            
            return wake_probability > WAKE_CONFIDENCE

    def run(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, 
                        rate=SAMPLE_RATE, input=True, 
                        frames_per_buffer=CHUNK_SIZE)
        
        print("\n" + "="*50)
        print("     VOICE ASSISTANT READY")
        print(f"     Status: WAITING FOR WAKE WORD...")
        print("="*50 + "\n")

        state = "WAITING_WAKE_WORD"

        try:
            while self.is_running:
                # Read audio chunk
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)

                # --- STATE 1: Waiting for Wake Word ---
                if state == "WAITING_WAKE_WORD":
                    is_detected = self._check_wake_word(data)
                    
                    if is_detected:
                        print("\n>>> 🟢 Wake Word Detected! Listening for command...")
                        # Switch state
                        state = "LISTENING_COMMAND"
                        # Reset STT buffer
                        self.recognizer.Reset()
                        # Clear audio buffer to prevent immediate re-trigger
                        self.audio_buffer.clear()

                # --- STATE 2: Listening for Command (STT) ---
                elif state == "LISTENING_COMMAND":
                    # Feed audio to Vosk
                    if self.recognizer.AcceptWaveform(data):
                        result = json.loads(self.recognizer.Result())
                        text = result.get("text", "")
                        
                        if text:
                            print(f">>> 🗣️ Heard: '{text}'")
                            
                            # --- Run NLU ---
                            nlu_res = self.nlu.parse(text)
                            
                            print(f"\n   [Analysis]")
                            print(f"   Intent:     {nlu_res['intent']}")
                            print(f"   Confidence: {nlu_res['confidence']:.1%}")
                            print(f"   Slots:      {nlu_res['slots']}")
                            print("-" * 30)
                            
                            print(">>> 💤 Back to sleep (Waiting for Wake Word)...\n")
                            state = "WAITING_WAKE_WORD"
                    else:
                        # Optional: Print partial results
                        # partial = json.loads(self.recognizer.PartialResult())
                        # print(f"\rListening... {partial.get('partial', '')}", end="")
                        pass

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.run()