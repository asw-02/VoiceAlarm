#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Routes German voice commands to safe local alarm-clock actions.

Qwen may suggest structured commands, but only this Python router validates
and hands confirmed payloads to the UI. Qwen never writes state.json directly.
"""

import json
import re
from datetime import datetime, timedelta

import requests

from config import Config, VoiceConfig
from voice_control.speech_format import date_to_german_words, time_to_german_words


ALLOWED_ACTIONS = {
    "set_alarm",
    "delete_alarm",
    "delete_all_alarms",
    "activate_alarm",
    "deactivate_alarm",
    "activate_all_alarms",
    "deactivate_all_alarms",
    "get_time",
    "get_date",
    "get_next_alarm",
    "get_alarm_count",
    "get_active_alarm_count",
    "has_alarm_tomorrow",
    "get_next_weekend_alarm",
    "list_active_alarms",
    "chat",
}


COMMAND_SYSTEM_PROMPT = """\
Du bist ein deutscher Sprachbefehl-Parser fuer einen Raspberry-Pi-Wecker.
Antworte ausschliesslich mit einem JSON-Objekt, ohne Markdown.

Erlaubte Aktionen:
- set_alarm
- delete_alarm
- delete_all_alarms
- activate_alarm
- deactivate_alarm
- activate_all_alarms
- deactivate_all_alarms
- get_time
- get_date
- get_next_alarm
- get_alarm_count
- get_active_alarm_count
- has_alarm_tomorrow
- get_next_weekend_alarm
- list_active_alarms
- chat

JSON-Felder:
- action: eine erlaubte Aktion
- hour: Integer 0-23 oder null
- minute: Integer 0-59 oder null
- days: Liste aus ["Mo","Di","Mi","Do","Fr","Sa","So"] oder null
- target_id: Integer oder null
- delete_all: Boolean

Regeln:
- Beachte typische Spracherkennungsfehler: "becker", "baecker", "wecka" oder
  aehnliche Woerter koennen "Wecker" bedeuten.
- "halb sieben" bedeutet 06:30, ausser abends/nachts deutet auf PM.
- "morgen" bedeutet nur der morgige Wochentag.
- "heute" bedeutet nur der heutige Wochentag.
- "werktags" oder "wochentags" bedeutet ["Mo","Di","Mi","Do","Fr"].
- "Wochenende" bedeutet ["Sa","So"].
- Wenn kein Tag genannt wird, setze days auf null.
- Fragen wie "welche Wecker sind aktiv", "welche Alarme sind an" oder
  "was ist gerade aktiv" bedeuten action "list_active_alarms".
- Fragen nach Anzahl bedeuten action "get_alarm_count".
- Allgemeine Gespraeche sind action "chat".
"""


class VoiceCommandRouter:
    """Executes simple local commands and safe alarm actions."""

    ALARM_WORDS = [
        "wecker",
        "weck",
        "alarm",
        "klingel",
        # Common Vosk confusions for "Wecker" / wake word bleed-through.
        "becker",
        "baecker",
        "bekker",
        "wecka",
        "wekker",
    ]

    SET_WORDS = ["stell", "stelle", "stellen", "einstellen", "setzen", "setze", "weck", "wecke"]

    DAY_NAMES = {
        "Mo": "Montag",
        "Di": "Dienstag",
        "Mi": "Mittwoch",
        "Do": "Donnerstag",
        "Fr": "Freitag",
        "Sa": "Samstag",
        "So": "Sonntag",
    }

    WEEKDAY_ALIASES = {
        "montag": "Mo",
        "dienstag": "Di",
        "mittwoch": "Mi",
        "donnerstag": "Do",
        "freitag": "Fr",
        "samstag": "Sa",
        "sonntag": "So",
    }

    NUMBER_WORDS = {
        "null": 0,
        "eins": 1,
        "ein": 1,
        "eine": 1,
        "einen": 1,
        "ersten": 1,
        "erste": 1,
        "zwei": 2,
        "zweiten": 2,
        "zweite": 2,
        "drei": 3,
        "dritten": 3,
        "dritte": 3,
        "vier": 4,
        "vierten": 4,
        "vierte": 4,
        "fuenf": 5,
        "fuenften": 5,
        "fuenfte": 5,
        "sechs": 6,
        "sechsten": 6,
        "sechste": 6,
        "sieben": 7,
        "siebten": 7,
        "siebte": 7,
        "acht": 8,
        "achten": 8,
        "achte": 8,
        "neun": 9,
        "neunten": 9,
        "neunte": 9,
        "zehn": 10,
        "elf": 11,
        "zwoelf": 12,
        "dreizehn": 13,
        "vierzehn": 14,
        "fuenfzehn": 15,
        "sechzehn": 16,
        "siebzehn": 17,
        "achtzehn": 18,
        "neunzehn": 19,
        "zwanzig": 20,
        "einundzwanzig": 21,
        "zweiundzwanzig": 22,
        "dreiundzwanzig": 23,
        "vierundzwanzig": 24,
        "fuenfundzwanzig": 25,
        "sechsundzwanzig": 26,
        "siebenundzwanzig": 27,
        "achtundzwanzig": 28,
        "neunundzwanzig": 29,
        "dreissig": 30,
        "einunddreissig": 31,
        "zweiunddreissig": 32,
        "dreiunddreissig": 33,
        "vierunddreissig": 34,
        "fuenfunddreissig": 35,
        "sechsunddreissig": 36,
        "siebenunddreissig": 37,
        "achtunddreissig": 38,
        "neununddreissig": 39,
        "vierzig": 40,
        "einundvierzig": 41,
        "zweiundvierzig": 42,
        "dreiundvierzig": 43,
        "vierundvierzig": 44,
        "fuenfundvierzig": 45,
        "sechsundvierzig": 46,
        "siebenundvierzig": 47,
        "achtundvierzig": 48,
        "neunundvierzig": 49,
        "fuenfzig": 50,
        "einundfuenfzig": 51,
        "zweiundfuenfzig": 52,
        "dreiundfuenfzig": 53,
        "vierundfuenfzig": 54,
        "fuenfundfuenfzig": 55,
        "sechsundfuenfzig": 56,
        "siebenundfuenfzig": 57,
        "achtundfuenfzig": 58,
        "neunundfuenfzig": 59,
        "sechzig": 60,
    }

    def __init__(self, ui, assistant):
        self.ui = ui
        self.assistant = assistant
        self.pending_command = None

    @staticmethod
    def _normalize(text):
        text = (text or "").lower().strip()
        replacements = {
            "\u00e4": "ae",
            "\u00f6": "oe",
            "\u00fc": "ue",
            "\u00df": "ss",
            "\u00e9": "e",
            "\u00e8": "e",
            "\u00e1": "a",
            "\u00e0": "a",
        }
        for wrong, correct in replacements.items():
            text = text.replace(wrong, correct)
        text = text.replace(".", ":")
        text = re.sub(r"[!,?;]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @classmethod
    def _numbers_to_digits(cls, text):
        words = []
        for word in text.split():
            clean = word.strip()
            words.append(str(cls.NUMBER_WORDS.get(clean, clean)))
        return " ".join(words)

    @classmethod
    def _has_alarm_word(cls, normalized_text):
        return any(word in normalized_text for word in cls.ALARM_WORDS)

    @classmethod
    def _has_set_word(cls, normalized_text):
        return any(word in normalized_text for word in cls.SET_WORDS)

    @classmethod
    def _has_day_hint(cls, normalized_text):
        return any(
            word in normalized_text
            for word in [
                "montag",
                "dienstag",
                "mittwoch",
                "donnerstag",
                "freitag",
                "samstag",
                "sonntag",
                "morgen",
                "heute",
                "uebermorgen",
                "wochenende",
                "werktags",
                "wochentags",
            ]
        )

    @classmethod
    def _has_time_hint(cls, normalized_text):
        text = cls._numbers_to_digits(normalized_text)
        if re.search(r"\b\d{1,2}:\d{1,2}\b", text):
            return True
        if re.search(r"\b\d{1,2}\s+uhr(?:\s+\d{1,2})?\b", text):
            return True
        if re.search(r"\b\d{1,2}\s+\d{1,2}\b", text):
            return True
        return any(
            word in normalized_text
            for word in [
                "uhr",
                "halb",
                "viertel",
                "dreiviertel",
                "vor",
                "nach",
                "minute",
                "minuten",
                "stunde",
                "stunden",
            ]
        )

    @staticmethod
    def _valid_time(hour, minute):
        return isinstance(hour, int) and isinstance(minute, int) and 0 <= hour <= 23 and 0 <= minute <= 59

    @staticmethod
    def _valid_days(days):
        return isinstance(days, list) and bool(days) and all(day in Config.DAYS for day in days)

    def reset_pending(self):
        self.pending_command = None

    def end_dialog(self, cancel_confirmation=True):
        self.reset_pending()
        if cancel_confirmation:
            self._cancel_voice_confirmation(show_message=False)

    def _alarm_context(self):
        alarms = self.ui.db.data.get("wecker", {})
        if not alarms:
            return "Keine Wecker gespeichert."

        lines = []
        for wid, alarm in sorted(alarms.items(), key=lambda item: int(item[0])):
            status = "aktiv" if alarm.get("active") else "inaktiv"
            days = ",".join(alarm.get("days", [])) or "keine Tage"
            lines.append(f"W{wid}: {alarm.get('time', '--:--')} ({days}) {status}")
        return "\n".join(lines)

    def _parse_with_qwen(self, text):
        now = self.ui.get_now()
        payload = {
            "model": VoiceConfig.OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": COMMAND_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Aktuelle Zeit: {now.strftime('%H:%M')}\n"
                        f"Heute: {Config.DAYS[now.weekday()]} {now.strftime('%d.%m.%Y')}\n"
                        f"Gespeicherte Wecker:\n{self._alarm_context()}\n\n"
                        f"Benutzer sagt: {text}"
                    ),
                },
            ],
            "stream": False,
            "think": False,
            "keep_alive": "5m",
            "options": {
                "num_ctx": 1024,
                "num_thread": 3,
                "num_predict": 160,
                "temperature": 0.0,
            },
        }

        try:
            response = self.assistant.session.post(
                VoiceConfig.OLLAMA_URL,
                json=payload,
                timeout=VoiceConfig.OLLAMA_TIMEOUT,
            )
            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "").strip()

            if "</think>" in content:
                content = content.split("</think>", 1)[-1].strip()

            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                content = match.group(0)

            return self._sanitize_command(json.loads(content))

        except (requests.RequestException, json.JSONDecodeError, ValueError, AttributeError) as exc:
            print(f"[CommandRouter] Qwen command parse failed: {exc}")
            return {"action": "chat"}

    def _sanitize_command(self, command):
        if not isinstance(command, dict):
            return {"action": "chat"}

        action = command.get("action")
        if action not in ALLOWED_ACTIONS:
            return {"action": "chat"}

        clean = {"action": action}
        for key in ("hour", "minute", "target_id"):
            value = command.get(key)
            if value is None or value == "":
                clean[key] = None
                continue
            try:
                clean[key] = int(value)
            except (TypeError, ValueError):
                clean[key] = None

        clean["days"] = self._sanitize_days(command.get("days"))
        clean["delete_all"] = bool(command.get("delete_all"))
        clean["date"] = command.get("date") if self._valid_date(command.get("date")) else None
        clean["needs_day_scope"] = bool(command.get("needs_day_scope"))
        clean["daypart_resolved"] = bool(command.get("daypart_resolved"))
        clean["source_text"] = command.get("source_text")
        return clean

    def _get_time(self):
        now = self.ui.get_now()
        return f"Es ist {time_to_german_words(now.hour, now.minute)}."

    def _get_date(self):
        now = self.ui.get_now()
        day_name = self.DAY_NAMES.get(Config.DAYS[now.weekday()], Config.DAYS[now.weekday()])
        return f"Heute ist {day_name}, {date_to_german_words(now)}."

    def _get_next_alarm(self):
        wid, alarm_time, days = self.ui.get_next_alarm_info()
        if wid == "Kein Wecker":
            return "Es ist kein aktiver Wecker gespeichert."
        return f"Der naechste Wecker ist {wid} um {self._time_text(alarm_time)}."

    def _get_alarm_count(self):
        alarms = self.ui.db.data.get("wecker", {})
        total_count = len(alarms)
        active_count = sum(1 for alarm in alarms.values() if alarm.get("active"))

        if total_count == 0:
            return "Du hast keine Wecker gespeichert."
        if active_count == 0:
            return f"Du hast {total_count} Wecker gespeichert, aber keiner ist aktiv."
        if active_count == 1:
            return f"Du hast {total_count} Wecker gespeichert. Davon ist einer aktiv."
        return f"Du hast {total_count} Wecker gespeichert. Davon sind {active_count} aktiv."

    def _get_active_alarm_count(self):
        active_count = sum(
            1 for alarm in self.ui.db.data.get("wecker", {}).values()
            if alarm.get("active")
        )

        if active_count == 0:
            return "Es ist kein Wecker aktiv."
        if active_count == 1:
            return "Ein Wecker ist aktiv."
        return f"{active_count} Wecker sind aktiv."

    @staticmethod
    def _parse_alarm_time(alarm):
        try:
            hour, minute = map(int, alarm.get("time", "").split(":"))
        except Exception:
            return None

        if VoiceCommandRouter._valid_time(hour, minute):
            return hour, minute
        return None

    @staticmethod
    def _time_text(time_value):
        try:
            hour, minute = map(int, str(time_value).split(":"))
        except Exception:
            return f"{time_value} Uhr"

        if not VoiceCommandRouter._valid_time(hour, minute):
            return f"{time_value} Uhr"
        return time_to_german_words(hour, minute)

    @staticmethod
    def _alarm_matches_date(alarm, date_value):
        if not alarm.get("active"):
            return False

        alarm_date = alarm.get("date")
        if alarm_date:
            return alarm_date == date_value.strftime("%Y-%m-%d")

        return Config.DAYS[date_value.weekday()] in (alarm.get("days") or [])

    def _active_alarms_for_date(self, date_value):
        return [
            (wid, alarm)
            for wid, alarm in sorted(self.ui.db.data.get("wecker", {}).items(), key=lambda item: int(item[0]))
            if self._alarm_matches_date(alarm, date_value)
        ]

    def _get_alarm_tomorrow(self):
        tomorrow = self.ui.get_now() + timedelta(days=1)
        alarms = self._active_alarms_for_date(tomorrow)
        day_name = self.DAY_NAMES.get(Config.DAYS[tomorrow.weekday()], Config.DAYS[tomorrow.weekday()])

        if not alarms:
            return f"Nein, fuer morgen ist kein aktiver Wecker gestellt."

        wid, alarm = alarms[0]
        time_text = self._time_text(alarm.get("time", "--:--"))
        if len(alarms) == 1:
            return f"Ja, morgen am {day_name} klingelt {wid} um {time_text}."

        return f"Ja, morgen am {day_name} sind {len(alarms)} Wecker gestellt. Der erste ist {wid} um {time_text}."

    def _get_next_weekend_alarm(self):
        now = self.ui.get_now()
        candidates = []

        for days_ahead in range(0, 15):
            date_value = now + timedelta(days=days_ahead)
            day_code = Config.DAYS[date_value.weekday()]
            if day_code not in ["Sa", "So"]:
                continue

            for wid, alarm in self._active_alarms_for_date(date_value):
                parsed_time = self._parse_alarm_time(alarm)
                if not parsed_time:
                    continue
                hour, minute = parsed_time
                candidate = date_value.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if candidate > now:
                    candidates.append((candidate, wid, alarm))

        if not candidates:
            return "Am Wochenende ist kein aktiver Wecker gestellt."

        candidate, wid, alarm = min(candidates, key=lambda item: item[0])
        day_name = self.DAY_NAMES.get(Config.DAYS[candidate.weekday()], Config.DAYS[candidate.weekday()])
        return f"Der naechste Wochenend-Wecker ist {wid} am {day_name} um {self._time_text(alarm.get('time', '--:--'))}."

    def _list_active_alarms(self):
        alarms = self.ui.db.data.get("wecker", {})
        active_alarms = [
            (wid, alarm)
            for wid, alarm in sorted(alarms.items(), key=lambda item: int(item[0]))
            if alarm.get("active")
        ]

        if not active_alarms:
            return "Es sind keine Wecker aktiv."

        if len(active_alarms) == 1:
            wid, alarm = active_alarms[0]
            return f"Aktiv ist Wecker {wid} um {self._time_text(alarm.get('time', '--:--'))}."

        parts = [
            f"Wecker {wid} um {self._time_text(alarm.get('time', '--:--'))}"
            for wid, alarm in active_alarms[:3]
        ]

        if len(active_alarms) > 3:
            remaining = len(active_alarms) - 3
            return f"Aktiv sind {', '.join(parts)} und {remaining} weitere."

        return f"Aktiv sind {', '.join(parts)}."

    @staticmethod
    def _valid_date(date_value):
        return isinstance(date_value, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", date_value) is not None

    def _format_date_label(self, date_value):
        try:
            alarm_date = datetime.strptime(date_value, "%Y-%m-%d")
        except Exception:
            return date_value

        day_name = self.DAY_NAMES.get(Config.DAYS[alarm_date.weekday()], Config.DAYS[alarm_date.weekday()])
        return f"Am {day_name}, {alarm_date.strftime('%d.%m.')}"

    def _format_alarm_label(self, days, date_value=None):
        if date_value:
            return self._format_date_label(date_value)
        if set(days) == set(Config.DAYS):
            return "Jeden Tag"
        if days == ["Mo", "Di", "Mi", "Do", "Fr"]:
            return "Werktags"
        if days == ["Sa", "So"]:
            return "Wochenende"
        if len(days) == 1:
            day_name = self.DAY_NAMES.get(days[0], days[0])
            return f"Jeden {day_name}"
        return ",".join(days)

    @staticmethod
    def _format_alarm_label_for_sentence(alarm_label):
        if alarm_label == "Wochenende":
            return "am Wochenende"
        return alarm_label[:1].lower() + alarm_label[1:]

    def _sanitize_days(self, days):
        if not isinstance(days, list):
            return None

        aliases = dict(self.WEEKDAY_ALIASES)
        aliases.update(
            {
                "mo": "Mo",
                "di": "Di",
                "mi": "Mi",
                "do": "Do",
                "fr": "Fr",
                "sa": "Sa",
                "so": "So",
            }
        )

        clean_days = []
        for day in days:
            normalized = self._normalize(str(day))
            mapped = aliases.get(normalized, str(day).strip())
            if mapped in Config.DAYS and mapped not in clean_days:
                clean_days.append(mapped)

        return sorted(clean_days, key=lambda value: Config.DAYS.index(value)) or None

    def _parse_days(self, normalized_text):
        now = self.ui.get_now()
        if any(word in normalized_text for word in ["jeden tag", "taeglich", "alle tage"]):
            return Config.DAYS[:]
        if any(word in normalized_text for word in ["werktags", "wochentags", "arbeitstage"]):
            return ["Mo", "Di", "Mi", "Do", "Fr"]
        if "wochenende" in normalized_text:
            return ["Sa", "So"]
        if "uebermorgen" in normalized_text:
            return [Config.DAYS[(now.weekday() + 2) % 7]]
        if "morgen" in normalized_text:
            return [Config.DAYS[(now.weekday() + 1) % 7]]
        if "heute" in normalized_text:
            return [Config.DAYS[now.weekday()]]

        days = [code for word, code in self.WEEKDAY_ALIASES.items() if word in normalized_text]
        return sorted(set(days), key=lambda value: Config.DAYS.index(value)) or None

    def _mentioned_weekdays(self, normalized_text):
        days = [
            code
            for word, code in self.WEEKDAY_ALIASES.items()
            if re.search(rf"\b{word}\b", normalized_text)
        ]
        return sorted(set(days), key=lambda value: Config.DAYS.index(value))

    @staticmethod
    def _has_explicit_day_scope(normalized_text):
        explicit_phrases = ["jeden tag"]
        explicit_words = [
            "heute",
            "morgen",
            "uebermorgen",
            "taeglich",
            "werktags",
            "wochentags",
            "wochenende",
            "diese",
            "diesen",
            "dieser",
            "diesem",
        ]

        return (
            any(phrase in normalized_text for phrase in explicit_phrases)
            or any(re.search(rf"\b{word}\b", normalized_text) for word in explicit_words)
            or "naechst" in normalized_text
            or "kommend" in normalized_text
        )

    def _weekday_date(self, day_code, hour, minute, scope):
        now = self.ui.get_now()
        target_index = Config.DAYS.index(day_code)
        days_ahead = (target_index - now.weekday()) % 7
        candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0) + timedelta(days=days_ahead)

        if scope == "next":
            candidate += timedelta(days=7)
        elif candidate <= now:
            candidate += timedelta(days=7)

        return candidate.strftime("%Y-%m-%d")

    @staticmethod
    def _explicit_weekday_scope(normalized_text):
        if "naechst" in normalized_text:
            return "next"
        weekday_pattern = r"(?:montag|dienstag|mittwoch|donnerstag|freitag|samstag|sonntag)"
        if any(word in normalized_text for word in ["diese", "diesen", "dieser", "diesem", "kommend"]):
            return "this"
        if re.search(rf"\bden\s+{weekday_pattern}\b", normalized_text):
            return "this"
        return None

    def _annotate_set_alarm_command(self, command, normalized_text):
        if command.get("action") != "set_alarm":
            return command

        days = self._sanitize_days(command.get("days"))
        hour = command.get("hour")
        minute = command.get("minute")

        if not self._valid_time(hour, minute) or not days or len(days) != 1:
            return command

        mentioned_days = self._mentioned_weekdays(normalized_text)
        if mentioned_days != days:
            return command

        command["source_text"] = normalized_text
        scope = self._explicit_weekday_scope(normalized_text)

        if scope:
            command["date"] = self._weekday_date(days[0], hour, minute, scope)
        elif not self._has_explicit_day_scope(normalized_text):
            command["needs_day_scope"] = True

        return command

    def _apply_daypart(self, hour, normalized_text):
        if hour is None:
            return None
        if any(word in normalized_text for word in ["nachmittag", "nachmittags", "abend", "abends"]) and 1 <= hour <= 11:
            return hour + 12
        if any(word in normalized_text for word in ["nacht", "nachts"]) and 6 <= hour <= 11:
            return hour + 12
        if any(word in normalized_text for word in ["morgen", "morgens", "frueh", "vormittag", "vormittags"]) and hour == 12:
            return 0
        return hour

    def _parse_relative_time(self, normalized_text):
        text = self._numbers_to_digits(normalized_text)
        now = self.ui.get_now()

        match = re.search(r"\bin\s+(\d{1,3})\s+minuten?\b", text)
        if match:
            future = now + timedelta(minutes=int(match.group(1)))
            return future.hour, future.minute, [Config.DAYS[future.weekday()]]

        match = re.search(r"\bin\s+(\d{1,2})\s+stunden?\b", text)
        if match:
            future = now + timedelta(hours=int(match.group(1)))
            return future.hour, future.minute, [Config.DAYS[future.weekday()]]

        if "in einer halben stunde" in normalized_text:
            future = now + timedelta(minutes=30)
            return future.hour, future.minute, [Config.DAYS[future.weekday()]]

        if "in einer stunde" in normalized_text:
            future = now + timedelta(hours=1)
            return future.hour, future.minute, [Config.DAYS[future.weekday()]]

        return None

    def _parse_time(self, normalized_text):
        relative = self._parse_relative_time(normalized_text)
        if relative:
            hour, minute, days = relative
            return hour, minute, days

        text = self._numbers_to_digits(normalized_text)
        patterns = [
            (r"\bhalb\s+(\d{1,2})\b", lambda m: ((int(m.group(1)) - 1) % 24, 30)),
            (r"\bviertel\s+nach\s+(\d{1,2})\b", lambda m: (int(m.group(1)), 15)),
            (r"\bviertel\s+vor\s+(\d{1,2})\b", lambda m: ((int(m.group(1)) - 1) % 24, 45)),
            (r"\bdreiviertel\s+(\d{1,2})\b", lambda m: ((int(m.group(1)) - 1) % 24, 45)),
            (r"\b(\d{1,2})\s+vor\s+(\d{1,2})\b", lambda m: ((int(m.group(2)) - 1) % 24, 60 - int(m.group(1)))),
            (r"\b(\d{1,2})\s+nach\s+(\d{1,2})\b", lambda m: (int(m.group(2)), int(m.group(1)))),
            (r"\b(\d{1,2}):(\d{1,2})\b", lambda m: (int(m.group(1)), int(m.group(2)))),
            (r"\b(\d{1,2})\s+uhr\s+(\d{1,2})\b", lambda m: (int(m.group(1)), int(m.group(2)))),
            (r"\b(?:um\s+)?(\d{1,2})\s+uhr\b", lambda m: (int(m.group(1)), 0)),
        ]

        for pattern, parser in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            hour, minute = parser(match)
            hour = self._apply_daypart(hour, normalized_text)
            if self._valid_time(hour, minute):
                return hour, minute, None

        match = re.search(r"\bum\s+(\d{1,2})\b", text)
        if match:
            hour = self._apply_daypart(int(match.group(1)), normalized_text)
            if self._valid_time(hour, 0):
                return hour, 0, None

        return None, None, None

    def _parse_target_id(self, normalized_text):
        text = self._numbers_to_digits(normalized_text)
        match = re.search(r"\b(?:wecker|alarm|nummer)\s+(\d{1,2})\b", text)
        if match:
            return int(match.group(1))
        return None

    def _find_alarm_id_by_time(self, hour, minute):
        if not self._valid_time(hour, minute):
            return None

        wanted_time = f"{hour:02d}:{minute:02d}"
        for wid, alarm in self.ui.db.data.get("wecker", {}).items():
            if alarm.get("time") == wanted_time:
                return wid
        return None

    def _find_next_alarm_id(self):
        next_info = self.ui.get_next_alarm_info()
        if next_info[0] != "Kein Wecker":
            return next_info[0].replace("W", "")
        return None

    @staticmethod
    def _has_daypart_hint(normalized_text):
        return any(
            word in normalized_text
            for word in [
                "morgens",
                "frueh",
                "vormittag",
                "vormittags",
                "nachmittag",
                "nachmittags",
                "abend",
                "abends",
                "nacht",
                "nachts",
            ]
        )

    @staticmethod
    def _needs_daypart_clarification(command):
        if command.get("daypart_resolved"):
            return False

        hour = command.get("hour")
        minute = command.get("minute")
        if not VoiceCommandRouter._valid_time(hour, minute):
            return False

        if not 1 <= hour <= 11:
            return False

        source_text = command.get("source_text") or ""
        return not VoiceCommandRouter._has_daypart_hint(source_text)

    def _request_target(self, action, base_command):
        self.pending_command = {"action": action, **base_command, "missing": "target"}
        return "Welchen Wecker meinst du?"

    def _request_time(self, base_command):
        self.pending_command = {"action": "set_alarm", **base_command, "missing": "time"}
        return "Um wie viel Uhr soll ich den Wecker stellen?"

    def _request_days(self, command, prompt="Jeden Tag?"):
        command = dict(command)
        command["missing"] = "days"
        self.pending_command = command
        return prompt

    def _request_daypart(self, command):
        command = dict(command)
        command["missing"] = "daypart"
        self.pending_command = command
        return "Meinst du morgens oder abends?"

    def _request_day_scope(self, command):
        command = dict(command)
        command["missing"] = "day_scope"
        command["needs_day_scope"] = False
        self.pending_command = command

        day = (command.get("days") or [None])[0]
        day_name = self.DAY_NAMES.get(day, "Wochentag")
        return f"Meinst du diesen {day_name} oder naechsten {day_name}?"

    @staticmethod
    def _parse_day_scope_choice(normalized_text):
        if "naechst" in normalized_text:
            return "next"
        if any(word in normalized_text for word in ["dies", "den", "heute", "kommend"]):
            return "this"
        return None

    @staticmethod
    def _parse_daypart_choice(normalized_text):
        if any(word in normalized_text for word in ["abend", "abends", "nachmittag", "nachmittags"]):
            return "pm"
        if any(word in normalized_text for word in ["morgen", "morgens", "frueh", "vormittag", "vormittags"]):
            return "am"
        return None

    def _execute_set_alarm(self, command):
        hour = command.get("hour")
        minute = command.get("minute")
        days = self._sanitize_days(command.get("days"))
        date_value = command.get("date")

        if hour is None or minute is None:
            return self._request_time({**command, "days": days})

        if not self._valid_time(hour, minute):
            return "Ich habe die Weckerzeit nicht sicher verstanden."

        if self._needs_daypart_clarification(command):
            return self._request_daypart({**command, "hour": hour, "minute": minute, "days": days})

        if not days:
            return self._request_days({**command, "hour": hour, "minute": minute})

        if not self._valid_days(days):
            return "Ich habe die Weckertage nicht sicher verstanden."

        if command.get("needs_day_scope"):
            return self._request_day_scope({**command, "days": days})

        time_str = f"{hour:02d}:{minute:02d}"
        payload = {
            "type": "create",
            "time": time_str,
            "days": days,
            "active": True,
        }
        if date_value:
            payload["date"] = date_value

        self.reset_pending()
        alarm_label = self._format_alarm_label(days, date_value)
        self.ui.request_confirmation(
            "Wecker stellen?",
            f"{alarm_label} {time_str}",
            payload,
        )

        if date_value or set(days) != set(Config.DAYS):
            spoken_label = self._format_alarm_label_for_sentence(alarm_label)
            return f"Soll ich den Wecker {spoken_label} um {time_to_german_words(hour, minute)} stellen?"

        return f"Soll ich den Wecker um {time_to_german_words(hour, minute)} stellen?"

    def _execute_delete_alarm(self, command):
        alarms = self.ui.db.data.get("wecker", {})
        target_id = command.get("target_id")

        if target_id is not None:
            target_id = str(target_id)
        else:
            target_id = self._find_alarm_id_by_time(command.get("hour"), command.get("minute"))

        if not target_id:
            return self._request_target("delete_alarm", command)

        if target_id not in alarms:
            return "Ich habe keinen passenden Wecker gefunden."

        alarm = alarms[target_id]
        days = alarm.get("days", [])
        days_label = self._format_alarm_label(days) if days else "Ohne Tage"
        self.reset_pending()
        self.ui.request_confirmation(
            f"Wecker {target_id} loeschen?",
            f"{alarm.get('time', '--:--')} {days_label}",
            {"type": "delete", "id": target_id},
        )
        return f"Soll ich Wecker {target_id} loeschen?"

    def _execute_delete_all(self):
        count = len(self.ui.db.data.get("wecker", {}))
        if count == 0:
            return "Es gibt keine Wecker zum Loeschen."

        self.reset_pending()
        self.ui.request_confirmation(
            "Alle loeschen?",
            f"{count} Wecker",
            {"type": "delete_all"},
        )
        return "Soll ich alle Wecker loeschen?"

    def _execute_set_alarm_active(self, command, active):
        alarms = self.ui.db.data.get("wecker", {})
        target_id = command.get("target_id")

        if target_id is not None:
            target_id = str(target_id)
        else:
            target_id = self._find_alarm_id_by_time(command.get("hour"), command.get("minute"))

        if not target_id:
            return self._request_target("activate_alarm" if active else "deactivate_alarm", command)

        if target_id not in alarms:
            return "Ich habe keinen passenden Wecker gefunden."

        alarm = alarms[target_id]
        action_text = "aktivieren" if active else "deaktivieren"
        self.reset_pending()
        self.ui.request_confirmation(
            f"Wecker {target_id} {action_text}?",
            f"{alarm.get('time', '--:--')} Uhr",
            {"type": "set_active", "id": target_id, "active": active},
        )
        return f"Soll ich Wecker {target_id} {action_text}?"

    def _execute_set_all_alarms_active(self, active):
        count = len(self.ui.db.data.get("wecker", {}))
        if count == 0:
            return "Es gibt keine Wecker zum Aendern."

        action_text = "aktivieren" if active else "deaktivieren"
        self.reset_pending()
        self.ui.request_confirmation(
            f"Alle {action_text}?",
            f"{count} Wecker",
            {"type": "set_all_active", "active": active},
        )
        return f"Soll ich alle Wecker {action_text}?"

    def _quick_local_action(self, normalized_text):
        if any(phrase in normalized_text for phrase in ["wie spaet", "wie viel uhr", "wieviel uhr", "uhrzeit"]):
            return "get_time"
        if any(phrase in normalized_text for phrase in ["welcher tag", "welchen tag", "datum", "heute fuer ein tag"]):
            return "get_date"
        if "morgen" in normalized_text and (
            "gestellt" in normalized_text
            or normalized_text.startswith("ist ")
            or normalized_text.startswith("sind ")
            or normalized_text.startswith("gibt ")
            or " gibt es " in normalized_text
        ):
            return "has_alarm_tomorrow"
        if "wochenende" in normalized_text and any(
            phrase in normalized_text
            for phrase in ["naechste wecker", "naechster wecker", "wann klingelt", "klingelt der naechste"]
        ):
            return "get_next_weekend_alarm"
        if any(
            phrase in normalized_text
            for phrase in [
                "wie viele wecker sind aktiv",
                "wieviele wecker sind aktiv",
                "wie viele aktive wecker",
                "wieviele aktive wecker",
                "wie viele alarm sind aktiv",
                "wieviele alarm sind aktiv",
            ]
        ):
            return "get_active_alarm_count"
        if any(
            phrase in normalized_text
            for phrase in [
                "wie viele wecker",
                "wieviele wecker",
                "anzahl wecker",
                "wecker gespeichert",
                "wie viele alarm",
                "wieviele alarm",
            ]
        ):
            return "get_alarm_count"
        if any(
            phrase in normalized_text
            for phrase in [
                "welche wecker sind aktiv",
                "welche wecker sind an",
                "aktive wecker",
                "wecker aktiv",
                "wecker sind aktiv",
                "sind wecker aktiv",
                "welche alarm sind aktiv",
                "aktive alarm",
            ]
        ):
            return "list_active_alarms"
        if any(phrase in normalized_text for phrase in ["naechste wecker", "naechster wecker", "wann klingelt"]):
            return "get_next_alarm"
        return None

    def _parse_local_alarm_command(self, normalized_text):
        delete_words = ["loesch", "loesche", "loeschen", "entfern", "entferne", "entfernen"]
        deactivate_words = ["deaktivier", "deaktiviere", "deaktivieren", "ausschalten", "ausmachen", "aus machen"]
        activate_words = ["aktivier", "aktiviere", "aktivieren", "einschalten", "anmachen", "an machen"]
        has_alarm_word = self._has_alarm_word(normalized_text)
        has_set_word = self._has_set_word(normalized_text)

        hour, minute, relative_days = self._parse_time(normalized_text)
        days = relative_days or self._parse_days(normalized_text)
        target_id = self._parse_target_id(normalized_text)

        has_time = self._valid_time(hour, minute)
        implicit_alarm_set = has_set_word and has_time

        if not has_alarm_word and not implicit_alarm_set:
            return None

        if "alle" in normalized_text and any(word in normalized_text for word in delete_words):
            return {"action": "delete_all_alarms", "delete_all": True}

        if any(word in normalized_text for word in deactivate_words) or (
            "schalte" in normalized_text and "aus" in normalized_text
        ) or (
            "mach" in normalized_text and "aus" in normalized_text
        ):
            if "alle" in normalized_text:
                return {"action": "deactivate_all_alarms"}
            return {"action": "deactivate_alarm", "target_id": target_id, "hour": hour, "minute": minute}

        if any(word in normalized_text for word in activate_words) or (
            "schalte" in normalized_text and ("ein" in normalized_text or "an" in normalized_text)
        ) or (
            "mach" in normalized_text and "an" in normalized_text
        ):
            if "alle" in normalized_text:
                return {"action": "activate_all_alarms"}
            return {"action": "activate_alarm", "target_id": target_id, "hour": hour, "minute": minute}

        if any(word in normalized_text for word in delete_words):
            return {"action": "delete_alarm", "target_id": target_id, "hour": hour, "minute": minute}

        if has_set_word:
            command = {
                "action": "set_alarm",
                "hour": hour,
                "minute": minute,
                "days": days,
                "daypart_resolved": bool(relative_days) or self._has_daypart_hint(normalized_text),
                "source_text": normalized_text,
            }
            return self._annotate_set_alarm_command(command, normalized_text)

        return None

    def _looks_like_alarm_command(self, normalized_text):
        command_words = [
            "loesch",
            "loesche",
            "loeschen",
            "entfern",
            "aktivier",
            "aktiviere",
            "aktivieren",
            "deaktivier",
            "deaktiviere",
            "deaktivieren",
            "schalte",
            "einschalten",
            "ausschalten",
        ] + self.SET_WORDS
        has_command_word = any(word in normalized_text for word in command_words)
        has_time_or_day = self._has_time_hint(normalized_text) or self._has_day_hint(normalized_text)

        if self._has_alarm_word(normalized_text) and (has_command_word or has_time_or_day):
            return True

        return self._has_set_word(normalized_text) and has_time_or_day

    def _should_try_qwen_intent(self, normalized_text):
        session = getattr(self.assistant, "session", None)
        if session is None:
            return False

        domain_words = [
            "alarme",
            "uhrzeit",
            "klingelt",
            "klingeln",
            "aktiv",
            "an",
            "aus",
        ]
        question_words = ["welche", "welcher", "welchen", "wann", "wie viele", "wieviele", "sind"]

        has_domain_word = self._has_alarm_word(normalized_text) or any(
            re.search(rf"\b{word}\b", normalized_text) for word in domain_words
        )
        has_question_shape = any(word in normalized_text for word in question_words)
        has_command_shape = (
            self._has_set_word(normalized_text)
            or any(word in normalized_text for word in ["loesch", "deaktivier", "aktivier", "schalte"])
        )

        return has_domain_word and (has_question_shape or has_command_shape or self._has_time_hint(normalized_text))

    def _handle_pending_response(self, normalized_text):
        if not self.pending_command:
            return None

        command = dict(self.pending_command)
        missing = command.pop("missing", None)

        if missing == "time":
            hour, minute, relative_days = self._parse_time(normalized_text)
            days = relative_days or self._parse_days(normalized_text) or command.get("days")
            command.update(
                {
                    "hour": hour,
                    "minute": minute,
                    "days": days,
                    "daypart_resolved": bool(relative_days) or self._has_daypart_hint(normalized_text),
                    "source_text": normalized_text,
                }
            )
            source_text = command.get("source_text") or normalized_text
            command = self._annotate_set_alarm_command(command, source_text)
            return self._dispatch(command)

        if missing == "daypart":
            choice = self._parse_daypart_choice(normalized_text)
            if not choice:
                return self._request_daypart(command)

            hour = command.get("hour")
            if choice == "pm" and isinstance(hour, int) and 1 <= hour <= 11:
                command["hour"] = hour + 12
            command["daypart_resolved"] = True
            return self._dispatch(command)

        if missing == "days":
            if self._is_yes(normalized_text):
                command["days"] = Config.DAYS[:]
                return self._dispatch(command)

            days = self._parse_days(normalized_text)
            if days:
                command["days"] = days
                return self._dispatch(command)

            return self._request_days(command, "An welchen Tagen soll der Wecker klingeln?")

        if missing == "target":
            target_id = self._parse_target_id(normalized_text)
            hour, minute, _ = self._parse_time(normalized_text)
            if "naechst" in normalized_text:
                target_id = self._find_next_alarm_id()
            command.update({"target_id": target_id, "hour": hour, "minute": minute})
            return self._dispatch(command)

        if missing == "day_scope":
            choice = self._parse_day_scope_choice(normalized_text)
            if not choice:
                return self._request_day_scope(command)

            days = self._sanitize_days(command.get("days")) or []
            if len(days) != 1:
                self.reset_pending()
                return "Ich habe den Wochentag nicht sicher verstanden."

            command["date"] = self._weekday_date(days[0], command.get("hour"), command.get("minute"), choice)
            command["needs_day_scope"] = False
            return self._dispatch(command)

        self.reset_pending()
        return None

    def _is_yes(self, normalized_text):
        return normalized_text in {"ja", "jawohl", "okay", "ok", "bestaetigen", "bestaetige", "passt", "mach das"}

    def _is_no(self, normalized_text):
        return normalized_text in {"nein", "nee", "no", "abbrechen", "abbruch", "stopp", "stop", "nicht"}

    def _cancel_voice_confirmation(self, show_message=True):
        if getattr(self.ui, "current_frame", None) not in {"voice_confirm", "overwrite_select"}:
            return False

        with self.ui.state_lock:
            self.ui.current_frame = "start"
            self.ui.voice_payload = None
            self.ui.voice_confirm_text = ("", "")

        if show_message:
            self.ui.show_temp_message("Abgebrochen", "")
        elif hasattr(self.ui, "draw"):
            self.ui.draw(force=True)

        return True

    def _handle_voice_confirmation(self, normalized_text):
        if getattr(self.ui, "current_frame", None) != "voice_confirm":
            return None

        if self._is_yes(normalized_text):
            self.reset_pending()
            self.ui.execute_voice_payload()
            return "Okay, ich habe es bestaetigt."

        if self._is_no(normalized_text):
            self.reset_pending()
            self._cancel_voice_confirmation()
            return "Okay, abgebrochen."

        return "Bitte sage ja zum Bestaetigen oder nein zum Abbrechen."

    def _dispatch(self, command):
        command = self._sanitize_command(command)
        action = command.get("action", "chat")

        if action == "get_time":
            return self._get_time()
        if action == "get_date":
            return self._get_date()
        if action == "get_next_alarm":
            return self._get_next_alarm()
        if action == "get_alarm_count":
            return self._get_alarm_count()
        if action == "get_active_alarm_count":
            return self._get_active_alarm_count()
        if action == "has_alarm_tomorrow":
            return self._get_alarm_tomorrow()
        if action == "get_next_weekend_alarm":
            return self._get_next_weekend_alarm()
        if action == "list_active_alarms":
            return self._list_active_alarms()
        if action == "delete_all_alarms" or command.get("delete_all"):
            return self._execute_delete_all()
        if action == "activate_all_alarms":
            return self._execute_set_all_alarms_active(True)
        if action == "deactivate_all_alarms":
            return self._execute_set_all_alarms_active(False)
        if action == "activate_alarm":
            return self._execute_set_alarm_active(command, True)
        if action == "deactivate_alarm":
            return self._execute_set_alarm_active(command, False)
        if action == "set_alarm":
            return self._execute_set_alarm(command)
        if action == "delete_alarm":
            return self._execute_delete_alarm(command)

        return None

    def handle(self, text):
        normalized = self._normalize(text)
        if not normalized:
            return None

        confirmation_answer = self._handle_voice_confirmation(normalized)
        if confirmation_answer is not None:
            return confirmation_answer

        if self.pending_command:
            return self._handle_pending_response(normalized)

        local_action = self._quick_local_action(normalized)
        local_command = self._parse_local_alarm_command(normalized)

        if local_action:
            command = {"action": local_action}
        elif local_command:
            command = local_command
        elif self._looks_like_alarm_command(normalized):
            command = self._parse_with_qwen(text)
            command["source_text"] = normalized
            command["daypart_resolved"] = self._has_daypart_hint(normalized)
            command = self._annotate_set_alarm_command(command, normalized)
        elif self._should_try_qwen_intent(normalized):
            command = self._parse_with_qwen(text)
            command["source_text"] = normalized
            command["daypart_resolved"] = self._has_daypart_hint(normalized)
            command = self._annotate_set_alarm_command(command, normalized)
        else:
            return None

        return self._dispatch(command)
