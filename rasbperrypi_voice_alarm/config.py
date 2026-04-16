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
    PIN_UP = 17
    PIN_DOWN = 27
    PIN_SAVE = 22
    PIN_AUTO = 5
    PIN_BUZZER = 12
    PIN_LIGHT_SENSOR = 23 
    PIN_MOSFET = 18
    PIN_LED_ROT = 6
    PIN_LED_GELB = 13
    PIN_LED_GRUEN = 19
    
    DAYS = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]

    @staticmethod
    def get_state_path() -> Path:
        """Returns the absolute path to the database JSON file."""
        return Path(__file__).resolve().parent / "state.json"

class VoiceConfig:
    """Configuration for Voice and AI models."""
    BASE_DIR = Path(__file__).resolve().parent

    NLU_MODEL_PATH = BASE_DIR / "models" / "nlu_model.onnx"
    STT_MODEL_PATH = BASE_DIR / "models" / "vosk-model-de-0.21"
    WAKE_MODEL_PATH = BASE_DIR / "models" / "wake_word_model.onnx"
    WAKE_STATS_PATH = BASE_DIR / "models" / "dataset_stats.pt"

    # Audio Settings
    MIC_SAMPLE_RATE = 44100   # Default microphone sample rate
    MODEL_SAMPLE_RATE = 16000 # Vosk/WakeWord standard rate
    CHUNK_SIZE = 4096         # Buffer for resampling
    WAKE_CONFIDENCE = 0.90
    SILENCE_THRESHOLD = 0.015