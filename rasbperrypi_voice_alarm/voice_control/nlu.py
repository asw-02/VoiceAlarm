#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Natural Language Understanding component.
Extracts intents and slots (time, weekdays, etc.) using an ONNX model.
"""

import re
import numpy as np
import onnxruntime as ort

class AlarmNLU:
    """Parses German text to identify alarm-related intents and parameters."""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.CONFIDENCE_THRESHOLD = 0.65
        
        # Load ONNX Session
        try:
            self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            self.label_name = self.session.get_outputs()[0].name
            self.proba_name = self.session.get_outputs()[1].name
        except Exception as e:
            print(f"[NLU Error] Failed to load ONNX: {e}")
            self.session = None

        # Slot Extraction Data (German mappings)
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
        
        self.relative_patterns = [
            (r"in\s+(\d+)\s+minuten?", lambda m: (0, int(m.group(1)))),
            (r"in\s+(\d+)\s+stunden?", lambda m: (int(m.group(1)), 0)),
            (r"in\s+einer\s+stunde", lambda m: (1, 0)),
            (r"in\s+einer\s+halben\s+stunde", lambda m: (0, 30))
        ]
        
        self.weekday_map = {
            "montag": 0, "dienstag": 1, "mittwoch": 2, "donnerstag": 3,
            "freitag": 4, "samstag": 5, "sonntag": 6,
            "heute": "PLUS_0", "morgen": "PLUS_1", "übermorgen": "PLUS_2"
        }
        
        self.daytime_rangers =  {
            "früh": (5, 10), "morgens": (6, 10), "vormittags": (9, 12),
            "mittags": (12, 13), "nachmittags": (13, 16), "abends": (16, 22),
            "nachts": (22, 5)
        }
        
    def _preprocess(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return " ".join(text.split())

    def _text_to_int(self, text: str):
        text = text.lower().strip()
        if text.isdigit(): return int(text)
        return self.word_to_num.get(text, None)
    
    def _convert_words_to_numbers(self, text: str) -> str:
        return " ".join(str(self._text_to_int(w) or w) for w in text.lower().split())

    def _extract_relative(self, text: str):
        text_proc = self._convert_words_to_numbers(text)
        for pattern, func in self.relative_patterns:
            m = re.search(pattern, text_proc)
            if m: return func(m) # Return (hours_delta, minutes_delta)
        return None

    def _extract_time(self, text: str):
        text_proc = " ".join(str(self._text_to_int(w) or w) for w in text.lower().split())
        for pattern, func in self.time_patterns:
            m = re.search(pattern, text_proc)
            if m: return func(m)
        return None, None

    def _extract_weekday(self, text: str):
        text = text.lower()
        for word, val in self.weekday_map.items():
            if word in text: return val
        return None

    def _extract_daytime(self, text: str):
        text = text.lower()
        for word, _ in self.daytime_rangers.items():
            if word in text: return word
        return None
    
    def _extract_id(self, text: str):
        """Searches for a simple digit (ID 1-9) in the text."""
        words = text.lower().split()
        for w in words:
            val = self._text_to_int(w)
            if val is not None and 1 <= val <= 9:
                return val
        return None   

    def _apply_daytime(self, hour, minute, daytime):
        if hour is None or daytime is None: return hour, minute
        start, end = self.daytime_rangers[daytime]
        if start > end:
            if not (hour >= start or hour < end): hour += 0
        else:
            if not (start <= hour < end): hour += 12
        return hour % 24, minute

    def parse(self, text: str) -> dict:
        """Parses user input into intents and parameters."""
        processed_input = self._preprocess(text)
        
        # ONNX Inference
        if self.session:
            input_data = np.array([[processed_input]], dtype=object)
            labels, probabilities = self.session.run(
                [self.label_name, self.proba_name], 
                {self.input_name: input_data}
            )
            predicted_intent = labels[0]
            max_proba = probabilities[0][predicted_intent]
        else:
            predicted_intent, max_proba = "unknown", 0.0

        final_intent = predicted_intent
        if max_proba < self.CONFIDENCE_THRESHOLD or predicted_intent == "no_intent":
            final_intent = "unknown"

        rel_time = self._extract_relative(text)
        hour, minute = self._extract_time(text)

        if hour is not None: hour = max(0, min(23, hour))
        if minute is not None: minute = max(0, min(59, minute))

        weekday = self._extract_weekday(text)
        daytime = self._extract_daytime(text)
        hour, minute = self._apply_daytime(hour, minute, daytime)
        target_id = self._extract_id(text)

        return {
            "text": text,
            "intent": final_intent,
            "confidence": float(max_proba),
            "slots": {
                "hour": hour, 
                "minute": minute, 
                "weekday": weekday,
                "relative_delta": rel_time, # (h, m) or None
                "target_id": target_id
            }
        }