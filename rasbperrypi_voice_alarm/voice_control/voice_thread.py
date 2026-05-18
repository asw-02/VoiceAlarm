#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Background thread for the German Qwen voice assistant.

Flow:
Wake word -> dynamic WAV recording -> Vosk transcription -> Qwen via Ollama
-> Piper speech output.
"""

import queue
import random
import threading
import time

import sounddevice as sd

from config import Config, VoiceConfig
from hardware.sensors import StatusLED
from voice_control.command_router import VoiceCommandRouter
from voice_control.qwen_assistant import QwenVoiceAssistant
from voice_control.wake_word_detection import WakeWordDetector


WAKE_PROMPTS = [
    "Was kann ich fuer dich tun?",
    "Wie kann ich dir helfen?",
    "Ja,Ich hoere dir zu.",
    "Sag mir, was ich tun soll.",
    "Ja, Ich bin bereit.",
    "Womit kann ich helfen?",
    "Was brauchst du gerade?",
    "Ja, so heiße ich, was kann ich für dich tun?"
]


class VoiceControlThread(threading.Thread):
    """Runs the wake-word gated Qwen assistant in the background."""

    def __init__(self, ui_reference):
        super().__init__(daemon=True)
        self.ui = ui_reference
        self._stop_event = threading.Event()
        self.audio_queue = queue.Queue(maxsize=20)
        self.leds = {}
        self._led_timers = []
        self.assistant = QwenVoiceAssistant()
        self.command_router = VoiceCommandRouter(self.ui, self.assistant)
        self.vosk_model = None
        self.wake_detector = None
        self.running = False
        self._alarm_voice_blocked = False

        self._init_leds()
        self._init_models()

    def _init_leds(self):
        try:
            self.leds = {
                "red": StatusLED(Config.PIN_LED_ROT),
                "yellow": StatusLED(Config.PIN_LED_GELB),
                "green": StatusLED(Config.PIN_LED_GRUEN),
            }
            self._all_leds_off()
        except Exception as exc:
            print(f"[Voice] LED init skipped: {exc}")
            self.leds = {}

    def _init_models(self):
        print("[Voice] Initializing German Qwen assistant...")
        self._show_temp_message("Sprachmodell", "wird geladen", duration=15)

        try:
            self.assistant.show_audio_devices()
        except Exception as exc:
            print(f"[Voice] Could not list audio devices: {exc}")

        try:
            self.vosk_model = self.assistant.load_vosk_model()
        except Exception as exc:
            print(f"[Voice] Fehler beim Laden des Vosk-Modells: {exc}")
            self._show_temp_message("Voice Error", "Vosk missing", duration=8)
            return

        try:
            self.wake_detector = WakeWordDetector()
        except Exception as exc:
            print(f"[Voice] Fehler beim Laden des Wake-Word-Modells: {exc}")
            self._show_temp_message("Voice Error", "Wake missing", duration=8)
            return

        try:
            sd.check_input_settings(
                device=VoiceConfig.MIC_DEVICE,
                channels=VoiceConfig.CHANNELS,
                samplerate=VoiceConfig.MIC_SAMPLE_RATE,
            )
            print("[Voice] Mikrofon unterstützt 48 kHz.")
        except Exception as exc:
            print("[Voice] Fehler: Mikrofon unterstützt diese Einstellungen nicht.")
            print(exc)
            self._show_temp_message("Voice Error", "Mic config", duration=8)
            return

        self.running = True
        self._show_temp_message("Voice Control", "bereit", duration=4)

    def stop(self):
        self._stop_event.set()
        self._cancel_led_timers()
        self._all_leds_off()
        self._end_voice_status()
        self._drain_audio_queue()
        if hasattr(self.assistant, "close"):
            self.assistant.close()

    def _show_temp_message(self, line1, line2="", duration=5):
        if not self.ui or not hasattr(self.ui, "show_temp_message"):
            return

        try:
            self.ui.show_temp_message(line1, line2, duration=duration)
            if hasattr(self.ui, "draw"):
                self.ui.draw(force=True)
        except Exception as exc:
            print(f"[Voice] UI message failed: {exc}")

    def _set_voice_status(self, line1, line2="", duration=None):
        if not self.ui:
            return

        try:
            if hasattr(self.ui, "set_voice_status"):
                self.ui.set_voice_status(line1, line2)
            else:
                self._show_temp_message(line1, line2, duration=duration or 5)
        except Exception as exc:
            print(f"[Voice] UI voice status failed: {exc}")

    def _end_voice_status(self):
        if not self.ui or not hasattr(self.ui, "end_voice_status"):
            return

        try:
            self.ui.end_voice_status()
        except Exception as exc:
            print(f"[Voice] UI voice status end failed: {exc}")

    def _all_leds_off(self):
        for led in self.leds.values():
            led.off()

    def _set_led(self, color, state):
        led = self.leds.get(color)
        if not led:
            return

        if state:
            led.on()
        else:
            led.off()

    def _blink_led_async(self, color, duration=2.0):
        self._set_led(color, True)
        self._led_timers = [timer for timer in self._led_timers if timer.is_alive()]
        timer = threading.Timer(duration, lambda: self._set_led(color, False))
        timer.daemon = True
        self._led_timers.append(timer)
        timer.start()

    def _cancel_led_timers(self):
        for timer in self._led_timers:
            timer.cancel()
        self._led_timers = []

    def _should_stop_voice(self):
        return self._stop_event.is_set() or self._alarm_is_ringing()

    def _cleanup(self):
        self._stop_event.set()
        self._cancel_led_timers()
        self._end_voice_status()
        self._drain_audio_queue()
        self._all_leds_off()

        for led in self.leds.values():
            if hasattr(led, "close"):
                led.close()
        self.leds = {}

        if hasattr(self.assistant, "close"):
            self.assistant.close()

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print("[Voice] Audio Status:", status)

        try:
            self.audio_queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass

    def _drain_audio_queue(self):
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    def _alarm_is_ringing(self):
        alarm_manager = getattr(self.ui, "alarm_manager", None)
        if not alarm_manager or not hasattr(alarm_manager, "is_ringing"):
            return False

        try:
            return alarm_manager.is_ringing()
        except Exception:
            return False

    def _end_voice_for_alarm(self):
        self._cancel_led_timers()
        self._all_leds_off()
        self.command_router.end_dialog(cancel_confirmation=True)
        self.assistant.reset_conversation()
        self._drain_audio_queue()
        self._end_voice_status()
        if not self._alarm_voice_blocked:
            self._show_temp_message("Alarm aktiv", "Voice aus", duration=2)
        self._alarm_voice_blocked = True

    def _pause_while_alarm_rings(self):
        if not self._alarm_is_ringing():
            self._alarm_voice_blocked = False
            return False

        self._end_voice_for_alarm()
        while not self._stop_event.is_set() and self._alarm_is_ringing():
            self._drain_audio_queue()
            time.sleep(0.5)

        self._alarm_voice_blocked = False
        self._drain_audio_queue()
        return True

    def _wait_for_wake_word(self):
        self._drain_audio_queue()

        if self._pause_while_alarm_rings():
            return False

        with sd.InputStream(
            device=VoiceConfig.MIC_DEVICE,
            channels=VoiceConfig.CHANNELS,
            samplerate=VoiceConfig.MIC_SAMPLE_RATE,
            blocksize=VoiceConfig.BLOCK_SIZE,
            dtype="float32",
            callback=self._audio_callback,
        ):
            while not self._stop_event.is_set():
                if self._alarm_is_ringing():
                    self._end_voice_for_alarm()
                    return False

                try:
                    audio = self.audio_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                triggered, wake_prob, not_wake_prob, hits = (
                    self.wake_detector.process_audio_block(audio)
                )

                print(
                    f"Wake: {wake_prob:.3f} | "
                    f"Not Wake: {not_wake_prob:.3f} | "
                    f"Hits: {hits}",
                    end="\r",
                )

                if triggered:
                    print("\nWake Word erkannt!\n")
                    return True

        return False

    def _handle_wake_word(self):
        if self._alarm_is_ringing():
            self._end_voice_for_alarm()
            return

        self._set_voice_status("Sprachsteuerung", "aktiv")
        self._blink_led_async("green", 1)

        if hasattr(self.ui, "last_button_time"):
            self.ui.last_button_time = time.time()

        self.assistant.reset_conversation()
        self.command_router.reset_pending()

        prompt = random.choice(WAKE_PROMPTS)
        self.assistant.speak(prompt, should_stop=self._should_stop_voice)

        if self._alarm_is_ringing():
            self._end_voice_for_alarm()
            return

        while not self._stop_event.is_set() and not self._alarm_is_ringing():
            self._set_led("yellow", True)
            user_text = self.assistant.listen_once_from_wav(
                self.vosk_model,
                stop_event=self._stop_event,
                status_callback=self._set_voice_status,
                should_stop=self._should_stop_voice,
            )
            self._set_led("yellow", False)

            if self._stop_event.is_set():
                self._end_voice_status()
                return

            if self._alarm_is_ringing():
                self._end_voice_for_alarm()
                return

            if not self.assistant.is_valid_speech(user_text):
                if user_text:
                    print(f"Ignoriert: {user_text}")
                    self._blink_led_async("red", 1.0)
                self.command_router.end_dialog(cancel_confirmation=True)
                self.assistant.reset_conversation()
                self._set_voice_status("Sprachsteuerung", "beendet")
                time.sleep(0.3)
                self._end_voice_status()
                return

            if hasattr(self.ui, "last_button_time"):
                self.ui.last_button_time = time.time()

            if self._alarm_is_ringing():
                self._end_voice_for_alarm()
                return

            answer = self.command_router.handle(user_text)

            if answer is None:
                if self._alarm_is_ringing():
                    self._end_voice_for_alarm()
                    return
                print("Qwen denkt nach...")
                self._set_voice_status("Wecker", "denkt nach")
                answer = self.assistant.ask_qwen(user_text)
            else:
                print("Sprachbefehl lokal verarbeitet.")
                if getattr(self.ui, "current_frame", None) == "voice_confirm":
                    pass
                else:
                    self._set_voice_status("Befehl", "erkannt")

            if self._alarm_is_ringing():
                self._end_voice_for_alarm()
                return

            print(f"Antwort: {answer}")
            self._set_voice_status(answer[:20], answer[20:40])
            self.assistant.speak(answer, should_stop=self._should_stop_voice)
            if self._alarm_is_ringing():
                self._end_voice_for_alarm()
                return
            self._blink_led_async("green", 1.0)
            time.sleep(0.5)

        if self._alarm_is_ringing():
            self._end_voice_for_alarm()
            return

        self._end_voice_status()

    def run(self):
        if not self.running:
            print("[Voice] Thread not started because initialization failed.")
            self._cleanup()
            return

        print("[Voice] Thread started. Waiting for Wake Word...")

        try:
            while not self._stop_event.is_set():
                try:
                    if self._pause_while_alarm_rings():
                        continue
                    if self._wait_for_wake_word():
                        self._handle_wake_word()
                except Exception as exc:
                    print(f"\n[Voice] Fehler: {exc}")
                    self._blink_led_async("red", 1.0)
                    self._end_voice_status()
                    self._show_temp_message("Voice Error", "siehe Konsole", duration=5)
                    time.sleep(0.5)
        finally:
            self._cleanup()
            print("[Voice] Thread stopped.")
