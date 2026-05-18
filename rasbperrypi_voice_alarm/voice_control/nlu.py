#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Deprecated compatibility wrapper for the old AlarmNLU import path.

The project now uses VoiceCommandRouter plus Qwen JSON parsing instead of the
previous ONNX NLU model. This facade remains only so older imports keep working.
"""

from voice_control.qwen_assistant import QwenVoiceAssistant


class AlarmNLU:
    """Legacy facade that forwards text to Qwen and returns a chat result."""

    def __init__(self, *args, **kwargs):
        self.assistant = QwenVoiceAssistant()

    def parse(self, text: str, alarm_data: dict = None) -> dict:
        answer = self.assistant.ask_qwen(text)
        return {
            "text": text,
            "intent": "chat",
            "confidence": 1.0,
            "response": answer,
            "slots": {},
        }
