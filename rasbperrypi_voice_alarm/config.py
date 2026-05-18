#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration constants, paths, and hardware pin mappings.
"""

from pathlib import Path

# --- Hardware Check for I2C/LCD ---
try:
    import smbus
    SMBUS_AVAILABLE = True
except ImportError:
    smbus = None
    SMBUS_AVAILABLE = False

try:
    from RPLCD.i2c import CharLCD
    USE_LCD = True
except ImportError:
    USE_LCD = False

class Config:
    """General system and hardware configuration."""
    VISIBLE_LINES = 4
    LCD_COLS = 20
    MAX_WECKER = 5  # Max number of alarms
    BOUNCE_TIME = 0.1

    # PIN MAPPING (Raspberry Pi GPIO)
    PIN_MENU = 4
    PIN_UP = 27
    PIN_DOWN = 17
    PIN_SAVE = 22
    PIN_AUTO = 5
    PIN_BUZZER = 12
    PIN_LIGHT_SENSOR = 23 
    PIN_MOSFET = 18
    PIN_LED_ROT = 6
    PIN_LED_GELB = 0
    PIN_LED_GRUEN = 11
    
    DAYS = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]

    @staticmethod
    def get_state_path() -> Path:
        """Returns the absolute path to the database JSON file."""
        return Path(__file__).resolve().parent / "state.json"

class VoiceConfig:
    """Configuration for the German Qwen voice assistant."""

    # Ollama / Qwen
    OLLAMA_URL = "http://localhost:11434/api/chat"
    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_MODEL = "qwen3:1.7b"
    OLLAMA_TIMEOUT = 120

    # Vosk / microphone
    VOSK_MODEL_PATH = "/home/oemer/vosk-stt/vosk-model-de-0.21"
    STT_MODEL_PATH = VOSK_MODEL_PATH

    MIC_DEVICE = 0
    SAMPLE_RATE = 48000
    MIC_SAMPLE_RATE = SAMPLE_RATE
    MODEL_SAMPLE_RATE = 16000
    CHANNELS = 1
    OUTPUT_WAV = "/tmp/vosk_input.wav"

    # Dynamic recording detection
    START_RMS = 350
    STOP_RMS = 200
    SILENCE_SECONDS = 3.0
    LISTEN_TIMEOUT_SECONDS = 5.0
    MIN_RECORD_SECONDS = 0.8
    BLOCK_DURATION = 0.1
    BLOCKSIZE = int(SAMPLE_RATE * BLOCK_DURATION)
    BLOCK_SIZE = BLOCKSIZE
    CHUNK_SIZE = BLOCKSIZE

    # Piper
    PIPER_BIN = "/home/oemer/piper-tts/piper/piper"
    PIPER_MODEL = "/home/oemer/piper-tts/de_DE-thorsten-medium.onnx"
    TTS_MODEL_PATH = PIPER_MODEL

    # Wake word model
    WAKE_MODEL_PATH = "/home/oemer/wake-word-detection/wake_word_model.onnx"
    WAKE_STATS_PATH = "/home/oemer/wake-word-detection/dataset_stats.pt"
    WAKE_CLASS_INDEX = 0
    WAKE_CONFIDENCE = 0.95
    SILENCE_THRESHOLD = 0.01
    MEL_TIME_FRAMES = 110
    REQUIRED_HITS = 3
    COOLDOWN_SECONDS = 3.0
