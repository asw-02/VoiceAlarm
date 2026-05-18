#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wake-word detection using the provided ONNX model and dataset statistics.
"""

import os
import queue
import time

import numpy as np
import onnxruntime as ort
import sounddevice as sd
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

from config import VoiceConfig


class WakeWordDetector:
    """Detects the configured wake word from 48 kHz microphone blocks."""

    def __init__(self):
        self.device = torch.device("cpu")

        print("Lade Wake-Word-Modell...")
        self.ww_session = ort.InferenceSession(
            VoiceConfig.WAKE_MODEL_PATH,
            providers=["CPUExecutionProvider"],
        )

        self.ww_input = self.ww_session.get_inputs()[0].name
        print("ONNX Input Name:", self.ww_input)

        print("Lade MelSpectrogram...")
        self.mel_transform = T.MelSpectrogram(
            sample_rate=VoiceConfig.MODEL_SAMPLE_RATE,
            n_fft=400,
            hop_length=160,
            n_mels=64,
        ).to(self.device)

        if os.path.exists(VoiceConfig.WAKE_STATS_PATH):
            print("Lade Normalisierungswerte...")
            stats = torch.load(VoiceConfig.WAKE_STATS_PATH, map_location=self.device)
            self.norm_mean = stats["mean"]
            self.norm_std = stats["std"]
        else:
            print("Keine dataset_stats.pt gefunden. Nutze mean=0, std=1.")
            self.norm_mean = torch.tensor(0.0)
            self.norm_std = torch.tensor(1.0)

        self.ww_buffer = torch.zeros(int(VoiceConfig.MODEL_SAMPLE_RATE * 1.1))
        self.audio_queue = queue.Queue()
        self.last_trigger_time = 0
        self.hit_count = 0

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print("Audio Status:", status)

        audio = indata[:, 0].copy()
        self.audio_queue.put(audio)

    def check_wake_word(self, audio_16k_tensor):
        rms = torch.sqrt(torch.mean(audio_16k_tensor**2))

        if rms.item() < VoiceConfig.SILENCE_THRESHOLD:
            return False, 0.0, 0.0

        new_samples = audio_16k_tensor.numel()

        if new_samples >= self.ww_buffer.numel():
            self.ww_buffer = audio_16k_tensor[-self.ww_buffer.numel():]
        else:
            self.ww_buffer = torch.roll(self.ww_buffer, -new_samples)
            self.ww_buffer[-new_samples:] = audio_16k_tensor

        mel = self.mel_transform(self.ww_buffer)

        mel = torchaudio.functional.amplitude_to_DB(
            mel,
            multiplier=10.0,
            amin=1e-10,
            db_multiplier=0.0,
        )

        mel = mel.unsqueeze(0).unsqueeze(0)
        mel = F.interpolate(
            mel,
            size=(64, VoiceConfig.MEL_TIME_FRAMES),
            mode="bilinear",
            align_corners=False,
        )

        mel = mel.squeeze(0).squeeze(0)
        mel = (mel - self.norm_mean) / (self.norm_std + 1e-6)

        input_np = mel.unsqueeze(0).unsqueeze(0).cpu().numpy().astype(np.float32)
        output = self.ww_session.run(None, {self.ww_input: input_np})

        logits = output[0]
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        wake_probability = float(probs[0][VoiceConfig.WAKE_CLASS_INDEX])
        not_wake_index = 1 if VoiceConfig.WAKE_CLASS_INDEX == 0 else 0
        not_wake_probability = float(probs[0][not_wake_index])

        detected = (
            wake_probability > VoiceConfig.WAKE_CONFIDENCE
            and wake_probability > not_wake_probability + 0.30
        )

        return detected, wake_probability, not_wake_probability

    def process_audio_block(self, audio):
        audio_tensor = torch.from_numpy(audio).float()

        if audio_tensor.ndim > 1:
            audio_tensor = audio_tensor[:, 0]

        audio_16k = torchaudio.functional.resample(
            audio_tensor,
            orig_freq=VoiceConfig.MIC_SAMPLE_RATE,
            new_freq=VoiceConfig.MODEL_SAMPLE_RATE,
        )

        detected, wake_prob, not_wake_prob = self.check_wake_word(audio_16k)

        if detected:
            self.hit_count += 1
        else:
            self.hit_count = 0

        now = time.time()
        triggered = (
            self.hit_count >= VoiceConfig.REQUIRED_HITS
            and now - self.last_trigger_time > VoiceConfig.COOLDOWN_SECONDS
        )
        reported_hits = self.hit_count

        if triggered:
            self.last_trigger_time = now
            self.hit_count = 0

        return triggered, wake_prob, not_wake_prob, reported_hits

    def run(self):
        print("Wake-Word-Detection gestartet.")
        print("Mikrofon nimmt mit 48 kHz auf.")
        print("Audio wird intern auf 16 kHz resampled.")
        print(f"Verwendetes Mikrofon-Device: {VoiceConfig.MIC_DEVICE}")
        print("STRG+C zum Beenden.\n")

        try:
            sd.check_input_settings(
                device=VoiceConfig.MIC_DEVICE,
                channels=1,
                samplerate=VoiceConfig.MIC_SAMPLE_RATE,
            )
            print("Mikrofon unterstützt 48 kHz.\n")
        except Exception as exc:
            print("Fehler: Mikrofon unterstützt diese Einstellungen nicht.")
            print(exc)
            return

        with sd.InputStream(
            device=VoiceConfig.MIC_DEVICE,
            channels=1,
            samplerate=VoiceConfig.MIC_SAMPLE_RATE,
            blocksize=VoiceConfig.BLOCK_SIZE,
            dtype="float32",
            callback=self.audio_callback,
        ):
            while True:
                audio = self.audio_queue.get()
                triggered, wake_prob, not_wake_prob, hits = self.process_audio_block(audio)

                print(
                    f"Wake: {wake_prob:.3f} | "
                    f"Not Wake: {not_wake_prob:.3f} | "
                    f"Hits: {hits}",
                    end="\r",
                )

                if triggered:
                    print("\nWake Word erkannt!\n")


if __name__ == "__main__":
    app = WakeWordDetector()
    app.run()
