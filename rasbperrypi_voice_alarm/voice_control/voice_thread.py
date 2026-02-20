#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Background thread managing the microphone, wake word detection, and speech recognition.
"""

import os
import time
import json
import threading
import queue
import numpy as np

import sounddevice as sd
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import onnxruntime as ort
from vosk import Model, KaldiRecognizer
from datetime import timedelta

from config import Config, VoiceConfig
from voice_control.nlu import AlarmNLU
from voice_control.resampler import StatefulResampler
from hardware.sensors import StatusLED

class VoiceControlThread(threading.Thread):
    """Listens continuously to audio and dispatches commands to the UI."""
    
    def __init__(self, ui_reference):
        super().__init__()
        # Initialisiere die LEDs über den sauberen Wrapper
        self.leds = {
            "red":    StatusLED(Config.PIN_LED_ROT),
            "yellow": StatusLED(Config.PIN_LED_GELB),
            "green":  StatusLED(Config.PIN_LED_GRUEN)
        }
        self._all_leds_off()
        self.ui = ui_reference
        self.daemon = True
        self._stop_event = threading.Event()
        self.audio_queue = queue.Queue(maxsize=20)
        self.ww_counter = 0
        self.device = torch.device("cpu")
        
        print("[Voice] Initializing models...")
        self.ui.show_temp_message("Voice Control", "loading ...", duration=15)
        self.ui.draw(force=True)
        
        # 1. NLU
        self.nlu = AlarmNLU(VoiceConfig.NLU_MODEL_PATH)
        
        # 2. STT (Vosk)
        if not os.path.exists(VoiceConfig.STT_MODEL_PATH):
            print(f"[Error] Vosk model missing: {VoiceConfig.STT_MODEL_PATH}")
            self.running = False
        else:
            self.running = True
            self.stt_model = Model(str(VoiceConfig.STT_MODEL_PATH))
            self.recognizer = KaldiRecognizer(self.stt_model, VoiceConfig.MODEL_SAMPLE_RATE)

        # 3. Wake Word
        self.ww_session = ort.InferenceSession(VoiceConfig.WAKE_MODEL_PATH, providers=["CPUExecutionProvider"])
        self.ww_input = self.ww_session.get_inputs()[0].name
        
        # Audio Resampler (44.1k -> 16k)
        self.resampler = StatefulResampler( 
            orig_sr=VoiceConfig.MIC_SAMPLE_RATE,   
            target_sr=VoiceConfig.MODEL_SAMPLE_RATE, 
            overlap=256,
            device=self.device)
        
        self.mel_transform = T.MelSpectrogram(sample_rate=16000, n_fft=400, hop_length=160, n_mels=64).to(self.device)
        
        # Wake Word Stats normalization
        if os.path.exists(VoiceConfig.WAKE_STATS_PATH):
            stats = torch.load(VoiceConfig.WAKE_STATS_PATH, map_location=self.device)
            self.norm_mean, self.norm_std = stats["mean"], stats["std"]
        else:
            self.norm_mean, self.norm_std = torch.tensor(0.0), torch.tensor(1.0)

        # Buffer for Wake Word (rolling buffer)
        self.ww_buffer = torch.zeros(int(16000 * 1.1)) # 1.1s buffer at 16k

    def stop(self):
        self._stop_event.set()

    def _all_leds_off(self):
        for led in self.leds.values():
            led.off()

    def _set_led(self, color, state):
        if color in self.leds:
            if state: self.leds[color].on()
            else: self.leds[color].off()

    def _audio_callback(self, indata, frames, time_info, status):
        try:
            self.audio_queue.put_nowait(indata.copy())
        except queue.Full:
            pass

    def _check_wake_word(self, audio_16k_tensor):
        rms = torch.sqrt(torch.mean(audio_16k_tensor**2))
        if rms.item() < VoiceConfig.SILENCE_THRESHOLD:
            return False
            
        new_samples = audio_16k_tensor.numel()
        self.ww_buffer = torch.roll(self.ww_buffer, -new_samples)
        self.ww_buffer[-new_samples:] = audio_16k_tensor
        
        mel = self.mel_transform(self.ww_buffer)
        mel = torchaudio.functional.amplitude_to_DB(mel, multiplier=10., amin=1e-10, db_multiplier=0.0, top_db=80.)
        
        mel = mel.unsqueeze(0).unsqueeze(0) # [1, 1, 64, Time]
        mel = F.interpolate(mel, size=(64, 110), mode='bilinear', align_corners=False)
        mel = (mel.squeeze(0).squeeze(0) - self.norm_mean) / (self.norm_std + 1e-6)
        
        input_np = mel.unsqueeze(0).unsqueeze(0).numpy()
        out = self.ww_session.run(None, {self.ww_input: input_np})
        probs = np.exp(out[0]) / np.sum(np.exp(out[0]), axis=1, keepdims=True)
        return probs[0][0] > VoiceConfig.WAKE_CONFIDENCE

    def process_intent_for_confirmation(self, nlu_res):
        print("\n" + "="*40)
        print(">>> [DEBUG] --- START VOICE PROCESSING ---")
        
        intent = nlu_res.get("intent")
        slots = nlu_res.get("slots", {})
        full_text = nlu_res.get("text", "").lower()
        
        # ==========================
        # CASE 1: SET ALARM
        # ==========================
        if intent == "set_alarm":
            now = self.ui.get_now()
            rel_delta = slots.get("relative_delta")
            
            if rel_delta:
                h_delta, m_delta = rel_delta
                future_time = now + timedelta(hours=h_delta, minutes=m_delta)
                hour, minute = future_time.hour, future_time.minute
                day_str = Config.DAYS[future_time.weekday()]
                days_list = [day_str]
                print(f">>> [DEBUG] Relative: +{h_delta}h {m_delta}m -> {day_str} {hour}:{minute}")
            else:
                hour = slots.get("hour")
                minute = slots.get("minute") or 0

                if hour is None:
                    self.ui.show_temp_message("Error", "No time understood")
                    return
                
                wd_raw = slots.get("weekday")
                days_list = []
                
                if "wochenende" in full_text:
                    days_list = ["Sa", "So"]
                elif any(x in full_text for x in ["wochentags", "werktags", "arbeitstagen", "unter der woche"]):
                    days_list = ["Mo", "Di", "Mi", "Do", "Fr"]
                elif wd_raw is not None:
                    wd_idx = 0
                    if isinstance(wd_raw, int): wd_idx = wd_raw
                    elif wd_raw == "PLUS_0": wd_idx = now.weekday()
                    elif wd_raw == "PLUS_1": wd_idx = (now.weekday() + 1) % 7
                    elif wd_raw == "PLUS_2": wd_idx = (now.weekday() + 2) % 7
                    days_list = [Config.DAYS[wd_idx]]
                else:
                    days_list = Config.DAYS[:] 

            time_str = f"{hour:02d}:{minute:02d}"
            
            payload = {
                "type": "create",
                "time": time_str,
                "days": days_list,
                "active": True
            }
            
            if set(days_list) == set(Config.DAYS): msg_line1 = "Every Day"
            elif days_list == ["Sa", "So"]: msg_line1 = "Weekend"
            elif days_list == ["Mo", "Di", "Mi", "Do", "Fr"]: msg_line1 = "Weekdays"
            elif len(days_list) == 1: msg_line1 = f"On {days_list[0]}"
            else: msg_line1 = "Set days"
                
            msg_line2 = f"At: {time_str}?"
            
            if hasattr(self.ui, "request_confirmation"):
                self.ui.request_confirmation(msg_line1, msg_line2, payload)

        # ==========================
        # CASE 2: DELETE ALARM
        # ==========================
        elif intent == "delete_alarm":
            if any(w in full_text for w in ["alle", "alles", "sämtliche"]):
                count = len(self.ui.db.data["wecker"])
                if count == 0:
                    self.ui.show_temp_message("Info", "Nothing to delete")
                    return
                payload = { "type": "delete_all" }
                msg_line1 = "DELETE ALL?"
                msg_line2 = f"Are you sure? ({count})"
                if hasattr(self.ui, "request_confirmation"):
                    self.ui.request_confirmation(msg_line1, msg_line2, payload)
                return

            target_id = None
            if slots.get("target_id"):
                target_id = str(slots.get("target_id"))
            elif slots.get("hour") is not None:
                s_hour = slots.get("hour")
                s_min = slots.get("minute") or 0
                search_time = f"{s_hour:02d}:{s_min:02d}"
                for wid, data in self.ui.db.data["wecker"].items():
                    if data["time"] == search_time:
                        target_id = wid
                        break
                if not target_id:
                    self.ui.show_temp_message("Not found", f"No alarm {search_time}")
                    return
            else:
                next_info = self.ui.get_next_alarm_info() 
                if next_info[0] != "Kein Wecker":
                    target_id = next_info[0].replace("W", "")
                else:
                    self.ui.show_temp_message("Info", "No active alarm")
                    return

            if target_id and target_id in self.ui.db.data["wecker"]:
                w_data = self.ui.db.data["wecker"][target_id]
                days_str = ",".join(w_data["days"]) if w_data["days"] else "Once"
                if len(days_str) > 10: days_str = days_str[:10] + "."

                payload = { "type": "delete", "id": target_id }
                msg_line1 = f"Delete W{target_id}?"
                msg_line2 = f"{w_data['time']} ({days_str})"

                if hasattr(self.ui, "request_confirmation"):
                    self.ui.request_confirmation(msg_line1, msg_line2, payload)
            else:
                self.ui.show_temp_message("Error", f"ID {target_id} not found")
    
    def _blink_led_async(self, color, duration=2.0):
        self._set_led(color, True)
        threading.Timer(duration, lambda: self._set_led(color, False)).start()

    def run(self):
        print("[Voice] Thread started. Waiting for Wake Word...")
        if getattr(self, "running", True) == False:
            return
            
        state = "WAITING"
        
        with sd.InputStream(device=2,
                    samplerate=VoiceConfig.MIC_SAMPLE_RATE,
                    blocksize=VoiceConfig.CHUNK_SIZE,
                    channels=1, dtype='float32',
                    callback=self._audio_callback):
            
            while not self._stop_event.is_set():
                if self.ui.current_frame == "voice_confirm" or self.ui.temp_msg_content is not None:
                    while not self.audio_queue.empty():
                        try: self.audio_queue.get_nowait()
                        except queue.Empty: pass
                    time.sleep(0.2)
                    continue

                if self.ui.alarm_manager.is_ringing():
                    while not self.audio_queue.empty(): self.audio_queue.get()
                    time.sleep(0.5)
                    continue

                try:
                    audio_chunk = self.audio_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                chunk_tensor = torch.from_numpy(audio_chunk).squeeze()
                chunk_16k = self.resampler.process(chunk_tensor)
                
                if state == "WAITING":
                    self.ww_counter = (self.ww_counter + 1) % 2
                    if self.ww_counter != 0: continue
                    
                    if self._check_wake_word(chunk_16k):
                        print("\n>>> WAKE WORD DETECTED!")
                        self.ui.last_button_time = time.time()
                        self.ui.draw(force=True)
                        
                        self._blink_led_async("green", 0.5) 
                        self._set_led("yellow", True)        
                        
                        while not self.audio_queue.empty():
                            try: self.audio_queue.get_nowait()
                            except queue.Empty: pass

                        self.recognizer.Reset()
                        state = "LISTENING"
                        
                elif state == "LISTENING":
                    audio_int16 = (chunk_16k.clamp(-1.0, 1.0).mul(32767).short().numpy().tobytes())
                    
                    if self.recognizer.AcceptWaveform(audio_int16):
                        self._set_led("yellow", False)

                        try:
                            res = json.loads(self.recognizer.Result())
                            text = res.get("text", "")
                            print(f">>> Heard: {text}")
                             
                            if text:
                                nlu_res = self.nlu.parse(text)
                                print(f">>> Intent: {nlu_res['intent']} | Slots: {nlu_res['slots']}")
                                
                                if nlu_res.get("intent") != "unknown":
                                    self._blink_led_async("green", 2.0)
                                    self.process_intent_for_confirmation(nlu_res)
                                else:
                                    self._blink_led_async("red", 2.0)
                            else:
                                self._blink_led_async("red", 2.0)

                        except Exception as e:
                            print(f">>> [ERROR] Command processing failed: {e}")
                            self._blink_led_async("red", 1.0)
                        
                        while not self.audio_queue.empty():
                            try: self.audio_queue.get_nowait()
                            except queue.Empty: pass

                        state = "WAITING"