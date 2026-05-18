#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from datetime import datetime

from voice_control.speech_format import date_to_german_words, format_for_tts, make_reply_informal


class SpeechFormatTests(unittest.TestCase):
    def test_format_time_for_tts(self):
        self.assertEqual(
            format_for_tts("Es ist 22:01 Uhr."),
            "Es ist zweiundzwanzig Uhr eins.",
        )

    def test_format_common_alarm_times_for_tts(self):
        self.assertEqual(format_for_tts("Der Wecker klingelt um 06:30 Uhr."), "Der Wecker klingelt um halb sieben.")
        self.assertEqual(format_for_tts("Der Wecker klingelt um 06:15 Uhr."), "Der Wecker klingelt um viertel nach sechs.")
        self.assertEqual(format_for_tts("Der Wecker klingelt um 06:45 Uhr."), "Der Wecker klingelt um viertel vor sieben.")

    def test_date_to_german_words(self):
        self.assertEqual(
            date_to_german_words(datetime(2026, 5, 16)),
            "den sechzehnten Mai sechsundzwanzig",
        )

    def test_qwen_formal_repeat_fallback_becomes_informal(self):
        self.assertEqual(
            make_reply_informal("Ich verstehe nicht. K\u00f6nnen Sie bitte wiederholen?"),
            "Ich verstehe nicht. Kannst du es bitte wiederholen?",
        )

    def test_qwen_formal_phrases_become_informal(self):
        self.assertEqual(
            make_reply_informal("K\u00f6nnen Sie mir helfen? Ich kann Ihnen antworten."),
            "Kannst du mir helfen? Ich kann dir antworten.",
        )

    def test_qwen_formal_possessive_becomes_informal(self):
        self.assertEqual(
            make_reply_informal("Ihre Wecker sind aktiv."),
            "Deine Wecker sind aktiv.",
        )


if __name__ == "__main__":
    unittest.main()
