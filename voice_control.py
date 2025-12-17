import os
import sys
import json
import re
import joblib
import numpy as np
import pyaudio
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import onnxruntime as ort 

# STT Import
from vosk import Model, KaldiRecognizer

# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = "voice_control"
NLU_MODEL_PATH = os.path.join(BASE_DIR, "nlu", "nlu_model.pkl")
STT_MODEL_PATH = os.path.join(BASE_DIR, "speech_to_text", "vosk-model-de-0.21")

# WICHTIG: Hier jetzt die .onnx Datei laden!
WAKE_MODEL_PATH = os.path.join(BASE_DIR, "wake_word_detection", "wake_word_model.onnx")
WAKE_STATS_PATH = os.path.join(BASE_DIR, "wake_word_detection", "dataset_stats.pt")

SAMPLE_RATE = 16000
CHUNK_SIZE = 4000
BUFFER_DURATION = 1.1
WAKE_CONFIDENCE = 0.95
SILENCE_THRESHOLD = 0.01  

# ============================================================
# 2. NLU CLASS (Unverändert)
# ============================================================
class AlarmNLU:
    def __init__(self, model_path):
        self.model_path = model_path
        self.classifier = None
        self.CONFIDENCE_THRESHOLD = 0.65
        
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
            "früh": (5, 10), "morgens": (6, 10), "vormittags": (9, 12),
            "mittags": (12, 13), "nachmittags": (13, 16), "abends": (16, 22),
            "abend": (16, 22), "nachts": (22, 5), "nacht": (22, 5)
        }

        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            print(f"[NLU Error] Model file not found: {self.model_path}")
            sys.exit(1)
        try:
            self.classifier = joblib.load(self.model_path)
        except Exception as e:
            print(f"[NLU Error] Failed to load model: {e}")
            sys.exit(1)

    def _preprocess(self, text):
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return " ".join(text.split())

    def _text_to_int(self, text):
        text = text.lower().strip()
        if text.isdigit():
            return int(text)
        return self.word_to_num.get(text, None)
    
    def _extract_time(self, text):
        text_proc = " ".join(str(self._text_to_int(w) or w) for w in text.lower().split())
        for pattern, func in self.time_patterns:
            m = re.search(pattern, text_proc)
            if m: return func(m)
        return None, None

    def _extract_weekday(self, text):
        text = text.lower()
        for word, val in self.weekday_map.items():
            if word in text: return val
        return None

    def _extract_daytime(self, text):
        text = text.lower()
        for word, _ in self.daytime_rangers.items():
            if word in text: return word
        return None

    def _apply_daytime(self, hour, minute, daytime):
        if hour is None or daytime is None:
            return hour, minute
        start, end = self.daytime_rangers[daytime]
        if start > end:
            if not (hour >= start or hour < end): hour += 0
        else:
            if not (start <= hour < end): hour += 12
        return hour % 24, minute

    def parse(self, text):
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
# 3. MAIN ASSISTANT CLASS (Optimized with ONNX)
# ============================================================

class VoiceAssistant:
    def __init__(self):
        self.is_running = True
        
        # PyTorch Settings (Only needed for Feature Extraction now)
        torch.set_num_threads(1)
        self.device = torch.device("cpu")
        
        print(f"[System] Running on: {self.device}")

        # --- Load NLU ---
        self.nlu = AlarmNLU(NLU_MODEL_PATH)

        # --- Load STT (Vosk) ---
        if not os.path.exists(STT_MODEL_PATH):
            sys.exit(f"[STT Error] Vosk model not found at {STT_MODEL_PATH}")
        
        print(f"[STT] Loading Vosk model...")
        self.stt_model = Model(STT_MODEL_PATH)
        grammar = ["wecker", "alarm", "stellen", "löschen", "wetter", "uhrzeit", 
                   "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", 
                   "acht", "neun", "zehn", "zwanzig", "dreißig", "halb", "viertel", 
                   "nach", "vor", "uhr", "bitte", "morgen", "heute"]
        self.recognizer = KaldiRecognizer(self.stt_model, SAMPLE_RATE, json.dumps(grammar))

        # --- Load Wake Word (ONNX Runtime) ---
        print(f"[Wake Word] Loading ONNX model from {WAKE_MODEL_PATH}...")
        try:
            # Load ONNX Inference Session
            # We explicitly ask for CPU execution provider
            self.ort_session = ort.InferenceSession(WAKE_MODEL_PATH, providers=["CPUExecutionProvider"])
            
            # Get input/output names dynamically so we don't have to guess
            self.input_name = self.ort_session.get_inputs()[0].name
            self.output_name = self.ort_session.get_outputs()[0].name
            
            print(f"[Wake Word] ONNX Loaded. Input: '{self.input_name}', Output: '{self.output_name}'")
            
        except Exception as e:
            sys.exit(f"[Wake Word Error] Failed to load ONNX model: {e}")

        # Load Stats (Standard PyTorch load is fine here, it's just numbers)
        if os.path.exists(WAKE_STATS_PATH):
            stats = torch.load(WAKE_STATS_PATH, map_location=self.device)
            self.norm_mean = stats["mean"]
            self.norm_std = stats["std"]
        else:
            self.norm_mean = torch.tensor(0.0)
            self.norm_std = torch.tensor(1.0)

        # Audio Transforms (PyTorch is still good for this DSP part)
        self.mel_transform = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=400, hop_length=160, n_mels=64
        ).to(self.device)

        # Buffer
        self.buffer_len = int(SAMPLE_RATE * BUFFER_DURATION)
        self.audio_buffer = np.zeros(self.buffer_len, dtype=np.float32)

    def _process_audio_chunk(self, audio_bytes):
        """
        Processes audio and returns a numpy array ready for ONNX.
        Returns: Numpy Array [1, 1, 64, 110] or None
        """
        # 1. Convert bytes to float32
        new_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # 2. Silence Gate
        rms = np.sqrt(np.mean(new_data**2))
        if rms < SILENCE_THRESHOLD:
            self.audio_buffer[:-CHUNK_SIZE] = self.audio_buffer[CHUNK_SIZE:]
            self.audio_buffer[-CHUNK_SIZE:] = new_data
            return None

        # 3. Buffer Shift
        self.audio_buffer[:-CHUNK_SIZE] = self.audio_buffer[CHUNK_SIZE:]
        self.audio_buffer[-CHUNK_SIZE:] = new_data
        
        # 4. Feature Extraction (Spectrogram)
        # We perform this part in PyTorch because it's convenient, 
        # but we switch to Numpy immediately after.
        waveform = torch.from_numpy(self.audio_buffer)

        with torch.inference_mode():
            mel_spec = self.mel_transform(waveform)
            mel_spec = torchaudio.functional.amplitude_to_DB(
                mel_spec, multiplier=10.0, amin=1e-10, db_multiplier=0.0, top_db=80.0
            )

            # Resize/Interpolate
            mel_spec = mel_spec.unsqueeze(0).unsqueeze(0) # [1, 1, 64, Time]
            mel_spec = F.interpolate(mel_spec, size=(64, 110), mode='bilinear', align_corners=False)
            mel_spec = mel_spec.squeeze(0).squeeze(0) 

            # Normalize
            mel_spec = (mel_spec - self.norm_mean) / (self.norm_std + 1e-6)
            
            # --- CRITICAL STEP FOR ONNX ---
            # Convert PyTorch Tensor -> Numpy Array
            # Add Batch and Channel dims: [1, 1, 64, 110]
            onnx_input = mel_spec.unsqueeze(0).unsqueeze(0).numpy()
            
            return onnx_input

    def _check_wake_word(self, audio_data):
        # Get Numpy Array
        input_data = self._process_audio_chunk(audio_data)
        
        if input_data is None:
            return False

        # Run ONNX Inference
        # This is where the magic happens: No PyTorch overhead!
        outputs = self.ort_session.run(
            [self.output_name], 
            {self.input_name: input_data}
        )
        
        # outputs[0] is the logits array (e.g. [[-2.5, 3.1]])
        logits = outputs[0]
        
        # Apply Softmax manually (since ONNX output is usually raw logits)
        # Simple softmax implementation for numpy
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        
        # Index 0 = "wake_word" (based on your training labels logic)
        wake_probability = probs[0][0]
        
        return wake_probability > WAKE_CONFIDENCE

    def run(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, 
                        rate=SAMPLE_RATE, input=True, 
                        frames_per_buffer=CHUNK_SIZE)
        
        print("\n" + "="*50)
        print("     VOICE ASSISTANT READY (ONNX POWERED)")
        print(f"     Status: WAITING FOR WAKE WORD...")
        print("="*50 + "\n")

        state = "WAITING_WAKE_WORD"

        try:
            while self.is_running:
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)

                if state == "WAITING_WAKE_WORD":
                    if self._check_wake_word(data):
                        print("\n>>> 🟢 Wake Word Detected! Listening...")
                        state = "LISTENING_COMMAND"
                        self.recognizer.Reset()
                        self.audio_buffer.fill(0)

                elif state == "LISTENING_COMMAND":
                    if self.recognizer.AcceptWaveform(data):
                        result = json.loads(self.recognizer.Result())
                        text = result.get("text", "")
                        
                        if text:
                            print(f">>> 🗣️ Heard: '{text}'")
                            nlu_res = self.nlu.parse(text)
                            
                            print(f"\n   [Analysis]")
                            print(f"   Intent:     {nlu_res['intent']}")
                            print(f"   Confidence: {nlu_res['confidence']:.1%}")
                            print(f"   Slots:      {nlu_res['slots']}")
                            print("-" * 30)
                            
                            print(">>> 💤 Back to sleep...\n")
                            state = "WAITING_WAKE_WORD"
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.run()