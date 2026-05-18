#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
German voice assistant using dynamic WAV recording, Vosk, Qwen via Ollama,
and Piper for speech output.
"""

import json
import queue
import subprocess
import time
import wave
from pathlib import Path

import numpy as np
import requests
import sounddevice as sd
from vosk import KaldiRecognizer, Model

from config import VoiceConfig
from voice_control.speech_format import format_for_tts, make_reply_informal


SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "Du bist ein deutscher Sprachassistent fuer einen Wecker. "
        "Sprich den Nutzer immer mit du an, nie mit Sie. "
        "Antworte kurz, direkt und natuerlich in  maximal vier bis fünf Saetzen. "
        "Keine Lists, kein Markdown, keine Emojis. "
        "Wenn eine Anfrage unklar ist, antworte genau: "
        "'Ich verstehe nicht. Kannst du es bitte wiederholen?'"
    ),
}


class QwenVoiceAssistant:
    """Wraps Vosk transcription, Qwen chat, and Piper speech output."""

    FALSE_POSITIVE_PHRASES = {
        "nun",
        "nun einen",
        "nun eine",
        "einen",
        "eine",
        "und",
        "äh",
        "ähm",
        "hm",
        "hmm",
    }

    FILLER_WORDS = {
        "nun",
        "einen",
        "eine",
        "ein",
        "und",
        "äh",
        "ähm",
        "hm",
        "hmm",
    }

    def __init__(self):
        self.session = requests.Session()
        self.messages = [SYSTEM_MESSAGE.copy()]

    def reset_conversation(self):
        self.messages = [SYSTEM_MESSAGE.copy()]

    def close(self):
        self.session.close()

    def show_audio_devices(self):
        print("\n=== Verfügbare Audio-Geräte ===")
        print(sd.query_devices())
        print("\nDefault device:", sd.default.device)
        print("===============================\n")

    def load_vosk_model(self):
        print("--- Lade Vosk-Modell ---")
        return Model(VoiceConfig.VOSK_MODEL_PATH)

    @staticmethod
    def calculate_rms(audio_block):
        audio_float = audio_block.astype(np.float32)
        return float(np.sqrt(np.mean(audio_float**2)))

    @staticmethod
    def _send_status(status_callback, line1, line2="", duration=3):
        if status_callback:
            status_callback(line1, line2, duration)

    @staticmethod
    def normalize_text(text):
        text = text.lower().strip()

        replacements = {
            "stock": "stop",
            "stopp": "stop",
            "stob": "stop",
            "stap": "stop",
            "tschüs": "tschüss",
        }

        for wrong, correct in replacements.items():
            text = text.replace(wrong, correct)

        for char in [".", ",", "!", "?", ":", ";"]:
            text = text.replace(char, "")

        return text.strip()

    def is_valid_speech(self, text):
        if not text:
            return False

        text = self.normalize_text(text)
        words = text.split()

        if not words:
            return False

        if text in self.FALSE_POSITIVE_PHRASES:
            return False

        if all(word in self.FILLER_WORDS for word in words):
            return False

        return True

    def record_until_silence(
        self,
        filename,
        stop_event=None,
        status_callback=None,
        should_stop=None,
    ):
        print("\nNehme direkt auf...")
        self._send_status(status_callback, "Wecker", "hoert zu", VoiceConfig.LISTEN_TIMEOUT_SECONDS + 1)

        recorded_blocks = []
        speech_started = False
        input_queue = queue.Queue(maxsize=80)
        input_overflows = 0
        dropped_blocks = 0

        start_time = time.monotonic()
        speech_start_time = None
        last_voice_time = start_time
        noise_rms = min(VoiceConfig.START_RMS, VoiceConfig.STOP_RMS) / 2.0

        def audio_callback(indata, frames, time_info, status):
            nonlocal dropped_blocks, input_overflows

            if status:
                input_overflows += 1

            try:
                input_queue.put_nowait(indata.copy())
            except queue.Full:
                dropped_blocks += 1
                try:
                    input_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    input_queue.put_nowait(indata.copy())
                except queue.Full:
                    pass

        try:
            with sd.InputStream(
                device=VoiceConfig.MIC_DEVICE,
                samplerate=VoiceConfig.SAMPLE_RATE,
                channels=VoiceConfig.CHANNELS,
                dtype="int16",
                blocksize=VoiceConfig.BLOCKSIZE,
                latency="high",
                callback=audio_callback,
            ):
                while (
                    (stop_event is None or not stop_event.is_set())
                    and (should_stop is None or not should_stop())
                ):
                    try:
                        audio_block = input_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue

                    if input_overflows:
                        print(f"Audio-Overflow ({input_overflows}x)")
                        input_overflows = 0

                    if dropped_blocks:
                        print(f"Audio-Puffer voll, {dropped_blocks} Block/Bloecke verworfen.")
                        dropped_blocks = 0

                    rms = self.calculate_rms(audio_block.reshape(-1))
                    now = time.monotonic()
                    recorded_blocks.append(audio_block.copy())
                    start_threshold = max(VoiceConfig.START_RMS, noise_rms * 3.0)
                    stop_threshold = max(VoiceConfig.STOP_RMS, noise_rms * 1.8)

                    if not speech_started:
                        if rms >= start_threshold:
                            speech_started = True
                            speech_start_time = now
                            last_voice_time = now
                            print("Sprache erkannt, nehme auf...")
                            self._send_status(status_callback, "Sprache", "erkannt", 3)
                        else:
                            noise_rms = (noise_rms * 0.95) + (rms * 0.05)
                            if now - start_time >= VoiceConfig.LISTEN_TIMEOUT_SECONDS:
                                print("Keine Sprache erkannt.")
                                self._send_status(status_callback, "Sprachsteuerung", "beendet", 2)
                                return False
                            continue

                    if rms >= stop_threshold:
                        last_voice_time = now

                    silence_duration = now - last_voice_time
                    record_duration = now - speech_start_time

                    if (
                        silence_duration >= VoiceConfig.SILENCE_SECONDS
                        and record_duration >= VoiceConfig.MIN_RECORD_SECONDS
                    ):
                        print("Stille erkannt, Aufnahme stoppt.")
                        break

        except Exception as exc:
            print(f"Fehler bei der Aufnahme: {exc}")
            return False

        if should_stop is not None and should_stop():
            print("Aufnahme abgebrochen, Alarm ist aktiv.")
            self._send_status(status_callback, "Alarm aktiv", "Voice aus", 2)
            return False

        if not speech_started:
            print("Keine Sprache erkannt.")
            return False

        if not recorded_blocks:
            print("Keine Sprache erkannt.")
            return False

        audio = np.concatenate(recorded_blocks, axis=0)

        try:
            with wave.open(filename, "wb") as wf:
                wf.setnchannels(VoiceConfig.CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(VoiceConfig.SAMPLE_RATE)
                wf.writeframes(audio.tobytes())

            return True

        except Exception as exc:
            print(f"Fehler beim Speichern der WAV-Datei: {exc}")
            return False

    def transcribe_wav(self, filename, model):
        wav_path = Path(filename)

        if not wav_path.exists():
            print(f"Datei nicht gefunden: {filename}")
            return ""

        try:
            wf = wave.open(str(wav_path), "rb")
        except Exception as exc:
            print(f"Fehler beim Öffnen der WAV-Datei: {exc}")
            return ""

        recognizer = KaldiRecognizer(model, wf.getframerate())
        recognizer.SetWords(False)

        text_parts = []

        while True:
            data = wf.readframes(4000)

            if len(data) == 0:
                break

            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").strip()

                if text:
                    text_parts.append(text)

        final_result = json.loads(recognizer.FinalResult())
        final_text = final_result.get("text", "").strip()

        if final_text:
            text_parts.append(final_text)

        wf.close()
        return self.normalize_text(" ".join(text_parts).strip())

    def listen_once_from_wav(
        self,
        model,
        stop_event=None,
        status_callback=None,
        should_stop=None,
    ):
        success = self.record_until_silence(
            VoiceConfig.OUTPUT_WAV,
            stop_event=stop_event,
            status_callback=status_callback,
            should_stop=should_stop,
        )

        if not success:
            return ""

        print("Vosk erkennt aus WAV...")
        self._send_status(status_callback, "Vosk", "erkennt", 4)
        text = self.transcribe_wav(VoiceConfig.OUTPUT_WAV, model)

        if text:
            print(f"Verstanden: {text}")
        else:
            print("Keine Sprache erkannt.")

        return text

    def ask_qwen(self, user_text):
        self.messages.append({"role": "user", "content": user_text})

        while len(self.messages) > 5:
            self.messages.pop(1)

        payload = {
            "model": VoiceConfig.OLLAMA_MODEL,
            "messages": self.messages,
            "stream": False,
            "think": False,
            "keep_alive": "2m",
            "options": {
                "num_ctx": 768,
                "num_thread": 3,
                "num_predict": 120,
                "temperature": 0.2,
            },
        }

        try:
            response = self.session.post(
                VoiceConfig.OLLAMA_URL,
                json=payload,
                timeout=VoiceConfig.OLLAMA_TIMEOUT,
            )
            response.raise_for_status()

            data = response.json()
            answer = data.get("message", {}).get("content", "").strip()

            if "</think>" in answer:
                answer = answer.split("</think>", 1)[-1].strip()

            if not answer:
                print("Ollama hat leeren content zurückgegeben.")
                print(json.dumps(data, indent=2, ensure_ascii=False))
                answer = "Ich habe gerade keine passende Antwort erzeugt."

            answer = make_reply_informal(answer)
            self.messages.append({"role": "assistant", "content": answer})
            return answer

        except requests.exceptions.ConnectionError:
            print("\nOllama ist nicht erreichbar.")
            return "Entschuldigung, Ollama ist gerade nicht erreichbar."

        except requests.exceptions.Timeout:
            print("\nOllama hat zu lange gebraucht.")
            return "Entschuldigung, die Antwort hat zu lange gedauert."

        except Exception as exc:
            print(f"\nOllama Fehler: {exc}")
            return "Entschuldigung, ich konnte keine Verbindung zu Ollama herstellen."

    def speak(self, text, should_stop=None):
        if not text:
            return

        spoken_text = format_for_tts(text)
        piper = None
        aplay = None

        try:
            piper = subprocess.Popen(
                [
                    "nice",
                    "-n",
                    "15",
                    VoiceConfig.PIPER_BIN,
                    "--model",
                    VoiceConfig.PIPER_MODEL,
                    "--output-raw",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )

            aplay = subprocess.Popen(
                [
                    "aplay",
                    "-r",
                    "22050",
                    "-f",
                    "S16_LE",
                    "-t",
                    "raw",
                    "-",
                ],
                stdin=piper.stdout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            piper.stdin.write(spoken_text)
            piper.stdin.close()

            while True:
                piper_done = piper.poll() is not None
                aplay_done = aplay.poll() is not None

                if piper_done and aplay_done:
                    break

                if should_stop is not None and should_stop():
                    print("Sprachausgabe abgebrochen, Alarm ist aktiv.")
                    for process in [piper, aplay]:
                        if process and process.poll() is None:
                            process.kill()
                    return

                time.sleep(0.05)

        except Exception as exc:
            print(f"Fehler bei der Sprachausgabe: {exc}")

            for process in [piper, aplay]:
                if process:
                    try:
                        process.kill()
                    except Exception:
                        pass


def main():
    assistant = QwenVoiceAssistant()

    print("=" * 50)
    print("Sprachassistent mit Auto-WAV, Vosk, Qwen und Piper")
    print("=" * 50)

    assistant.show_audio_devices()

    try:
        model = assistant.load_vosk_model()
    except Exception as exc:
        print(f"Fehler beim Laden des Vosk-Modells: {exc}")
        return

    print("\nSprachassistent bereit!")
    print("STRG+C zum Beenden.")

    while True:
        try:
            user_text = assistant.listen_once_from_wav(model)

            if not assistant.is_valid_speech(user_text):
                if user_text:
                    print(f"Ignoriert: {user_text}")
                time.sleep(0.3)
                continue

            print("Qwen denkt nach...")
            answer = assistant.ask_qwen(user_text)

            print(f"Antwort: {answer}")
            assistant.speak(answer)
            time.sleep(0.5)

        except KeyboardInterrupt:
            print("\nBeendet durch Benutzer.")
            break

        except Exception as exc:
            print(f"\nFehler: {exc}")
            assistant.speak("Da ist ein technisches Problem aufgetreten.")
            time.sleep(0.5)


if __name__ == "__main__":
    main()
