#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Deprecated compatibility wrapper for Piper speech output.

New code should use QwenVoiceAssistant.speak() directly. This module remains so
older imports keep working while the assistant uses the external Piper binary
configured in VoiceConfig.
"""

import threading

from voice_control.qwen_assistant import QwenVoiceAssistant


class VoiceSynthesizer:
    """Speaks text asynchronously through the new Piper subprocess pipeline."""

    def __init__(self):
        self.assistant = QwenVoiceAssistant()
        self.is_playing = False

    def speak(self, text: str, on_complete=None):
        if not text or not text.strip():
            if on_complete:
                on_complete()
            return

        def _speak_thread():
            self.is_playing = True
            try:
                self.assistant.speak(text)
            finally:
                self.is_playing = False
                if on_complete:
                    on_complete()

        thread = threading.Thread(target=_speak_thread, daemon=True)
        thread.start()
