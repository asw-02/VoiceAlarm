#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import unittest
from datetime import datetime

from voice_control.command_router import VoiceCommandRouter


class FakeDB:
    def __init__(self):
        self.lock = threading.RLock()
        self.data = {
            "wecker": {
                "1": {"time": "07:00", "days": ["Mo", "Di"], "active": True},
                "2": {"time": "08:30", "days": ["Sa", "So"], "active": False},
            }
        }

    def save(self):
        pass


class FakeAlarmManager:
    def __init__(self):
        self.stopped = False

    def is_ringing(self):
        return False

    def stop_alarm(self):
        self.stopped = True


class FakeUI:
    def __init__(self):
        self.db = FakeDB()
        self.state_lock = threading.RLock()
        self.current_frame = "start"
        self.voice_payload = None
        self.voice_confirm_text = ("", "")
        self.temp_messages = []
        self.executed_payload = False
        self.alarm_manager = FakeAlarmManager()

    def get_now(self):
        return datetime(2026, 5, 10, 10, 0)

    def get_next_alarm_info(self):
        for wid, alarm in sorted(self.db.data["wecker"].items(), key=lambda item: int(item[0])):
            if alarm.get("active"):
                return f"W{wid}", alarm["time"], ",".join(alarm.get("days", []))
        return "Kein Wecker", "", ""

    def request_confirmation(self, line1, line2, payload):
        self.voice_confirm_text = (line1, line2)
        self.voice_payload = payload
        self.current_frame = "voice_confirm"

    def execute_voice_payload(self):
        self.executed_payload = True
        self.current_frame = "start"
        self.voice_payload = None

    def show_temp_message(self, line1, line2="", duration=3):
        self.temp_messages.append((line1, line2, duration))

    def draw(self, force=False):
        pass


class FakeAssistant:
    session = None


class FakeQwenResponse:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"message": {"content": self.content}}


class FakeQwenSession:
    def __init__(self, content):
        self.content = content

    def post(self, *args, **kwargs):
        return FakeQwenResponse(self.content)


class FakeQwenAssistant:
    def __init__(self, content):
        self.session = FakeQwenSession(content)


class CommandRouterTests(unittest.TestCase):
    def setUp(self):
        self.ui = FakeUI()
        self.router = VoiceCommandRouter(self.ui, FakeAssistant())

    def test_alarm_count(self):
        answer = self.router.handle("Wie viele Wecker habe ich gespeichert?")
        self.assertIn("2 Wecker", answer)
        self.assertIn("einer aktiv", answer)

    def test_active_alarm_count(self):
        answer = self.router.handle("Wie viele Wecker sind aktiv?")
        self.assertEqual(answer, "Ein Wecker ist aktiv.")
        self.assertIsNone(self.ui.voice_payload)

    def test_are_all_alarms_off(self):
        answer = self.router.handle("Sind alle Wecker aus?")
        self.assertEqual(answer, "Nein, ein Wecker ist aktiv.")
        self.assertIsNone(self.ui.voice_payload)

        self.ui.db.data["wecker"]["1"]["active"] = False

        answer = self.router.handle("Sind alle Wecker aus?")
        self.assertEqual(answer, "Ja, alle Wecker sind aus.")
        self.assertIsNone(self.ui.voice_payload)

    def test_get_alarm_by_id(self):
        answer = self.router.handle("Wann klingelt Wecker eins?")
        self.assertIn("Wecker 1 klingelt um sieben Uhr", answer)
        self.assertIn("aktiv", answer)
        self.assertIsNone(self.ui.voice_payload)

    def test_list_tomorrow_active_alarms(self):
        answer = self.router.handle("Welche Wecker sind morgen aktiv?")
        self.assertEqual(answer, "Morgen am Montag ist Wecker 1 um sieben Uhr aktiv.")
        self.assertIsNone(self.ui.voice_payload)

    def test_alarm_is_set_tomorrow(self):
        answer = self.router.handle("Ist morgen ein Wecker gestellt?")
        self.assertEqual(answer, "Ja, morgen am Montag klingelt 1 um sieben Uhr.")
        self.assertIsNone(self.ui.voice_payload)

    def test_next_weekend_alarm(self):
        self.ui.db.data["wecker"]["2"]["active"] = True

        answer = self.router.handle("Wann klingelt der naechste Wecker am Wochenende?")

        self.assertEqual(answer, "Der naechste Wochenend-Wecker ist 2 am Samstag um halb neun.")
        self.assertIsNone(self.ui.voice_payload)

    def test_list_active_alarms_with_inserted_gerade(self):
        assistant = FakeQwenAssistant(
            '{"action":"list_active_alarms","hour":null,"minute":null,'
            '"days":null,"target_id":null,"delete_all":false}'
        )
        router = VoiceCommandRouter(self.ui, assistant)

        answer = router.handle("welche alarm sind gerade aktiv")

        self.assertIn("Aktiv ist Wecker 1", answer)
        self.assertIn("sieben Uhr", answer)

    def test_get_date_uses_full_weekday_name(self):
        answer = self.router.handle("Welcher Tag ist heute?")
        self.assertEqual(answer, "Heute ist Sonntag, den zehnten Mai sechsundzwanzig.")
        self.assertNotIn("So,", answer)

    def test_set_alarm_halb_vier(self):
        answer = self.router.handle("Stell einen Wecker auf halb vier")
        self.assertEqual(answer, "Meinst du morgens oder abends?")
        self.assertIsNone(self.ui.voice_payload)

        answer = self.router.handle("morgens")
        self.assertEqual(answer, "Jeden Tag?")

        answer = self.router.handle("ja")

        self.assertIn("halb vier", answer)
        self.assertIn("Soll ich", answer)
        self.assertTrue(answer.endswith("?"))
        self.assertEqual(self.ui.voice_payload["type"], "create")
        self.assertEqual(self.ui.voice_payload["time"], "03:30")
        self.assertEqual(self.ui.voice_payload["days"], ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"])

    def test_ambiguous_alarm_time_can_be_set_to_evening(self):
        answer = self.router.handle("Stell den Alarm um halb sechs")
        self.assertEqual(answer, "Meinst du morgens oder abends?")

        answer = self.router.handle("abends")
        self.assertEqual(answer, "Jeden Tag?")

        answer = self.router.handle("ja")

        self.assertIn("halb sechs", answer)
        self.assertEqual(self.ui.voice_payload["time"], "17:30")

    def test_set_alarm_tomorrow_quarter_after_six(self):
        answer = self.router.handle("Stell morgen um viertel nach sechs einen Wecker")
        self.assertEqual(answer, "Meinst du morgens oder abends?")

        answer = self.router.handle("morgens")

        self.assertIn("viertel nach sechs", answer)
        self.assertEqual(self.ui.voice_payload["time"], "06:15")
        self.assertEqual(self.ui.voice_payload["days"], ["Mo"])

    def test_set_alarm_ten_before_seven(self):
        answer = self.router.handle("Stell einen Wecker auf zehn vor sieben")
        self.assertEqual(answer, "Meinst du morgens oder abends?")

        answer = self.router.handle("morgens")
        self.assertEqual(answer, "Jeden Tag?")

        answer = self.router.handle("nur dienstags")

        self.assertIn("sechs Uhr f\u00fcnfzig", answer)
        self.assertEqual(self.ui.voice_payload["time"], "06:50")
        self.assertEqual(self.ui.voice_payload["days"], ["Di"])
        self.assertIn("jeden Dienstag", answer)

    def test_set_alarm_weekend_followup(self):
        answer = self.router.handle("kannst du alarm um halb vier einstellen")
        self.assertEqual(answer, "Meinst du morgens oder abends?")

        answer = self.router.handle("morgens")
        self.assertEqual(answer, "Jeden Tag?")

        answer = self.router.handle("wochenende")

        self.assertIn("halb vier", answer)
        self.assertEqual(self.ui.voice_payload["days"], ["Sa", "So"])

    def test_set_alarm_from_vosk_becker_phrase(self):
        answer = self.router.handle(
            "katze mia becker morgen um sechs uhr drei\u00dfig einstellen"
        )
        self.assertEqual(answer, "Meinst du morgens oder abends?")

        answer = self.router.handle("morgens")

        self.assertIn("halb sieben", answer)
        self.assertEqual(self.ui.voice_payload["type"], "create")
        self.assertEqual(self.ui.voice_payload["time"], "06:30")
        self.assertEqual(self.ui.voice_payload["days"], ["Mo"])

    def test_qwen_intent_parser_sets_alarm_when_local_parse_is_unclear(self):
        assistant = FakeQwenAssistant(
            '{"action":"set_alarm","hour":6,"minute":30,'
            '"days":["Mo"],"target_id":null,"delete_all":false}'
        )
        router = VoiceCommandRouter(self.ui, assistant)

        answer = router.handle("becker fuer morgen sechs dreissig")
        self.assertEqual(answer, "Meinst du morgens oder abends?")

        answer = router.handle("morgens")

        self.assertIn("halb sieben", answer)
        self.assertEqual(self.ui.voice_payload["type"], "create")
        self.assertEqual(self.ui.voice_payload["time"], "06:30")
        self.assertEqual(self.ui.voice_payload["days"], ["Mo"])

    def test_bare_weekday_alarm_asks_for_day_scope(self):
        answer = self.router.handle("stell einen alarm um halb vier fuer dienstag")
        self.assertEqual(answer, "Meinst du morgens oder abends?")

        answer = self.router.handle("morgens")

        self.assertIn("diesen Dienstag", answer)
        self.assertIn("naechsten Dienstag", answer)
        self.assertIsNone(self.ui.voice_payload)

    def test_weekday_alarm_next_tuesday_followup_sets_date(self):
        self.router.handle("stell einen alarm um halb vier fuer dienstag")
        self.router.handle("morgens")

        answer = self.router.handle("naechsten dienstag")

        self.assertIn("halb vier", answer)
        self.assertEqual(self.ui.voice_payload["type"], "create")
        self.assertEqual(self.ui.voice_payload["time"], "03:30")
        self.assertEqual(self.ui.voice_payload["days"], ["Di"])
        self.assertEqual(self.ui.voice_payload["date"], "2026-05-19")

    def test_explicit_this_tuesday_sets_date(self):
        answer = self.router.handle("stell einen alarm um halb vier fuer diesen dienstag")
        self.assertEqual(answer, "Meinst du morgens oder abends?")

        answer = self.router.handle("morgens")

        self.assertIn("halb vier", answer)
        self.assertEqual(self.ui.voice_payload["type"], "create")
        self.assertEqual(self.ui.voice_payload["days"], ["Di"])
        self.assertEqual(self.ui.voice_payload["date"], "2026-05-12")

    def test_set_alarm_relative_minutes(self):
        answer = self.router.handle("Stell einen Wecker in zwanzig Minuten")
        self.assertIn("zehn Uhr zwanzig", answer)
        self.assertEqual(self.ui.voice_payload["time"], "10:20")
        self.assertEqual(self.ui.voice_payload["days"], ["So"])

    def test_set_alarm_relative_compound_minutes(self):
        answer = self.router.handle("Stell einen Wecker in einundzwanzig Minuten")
        self.assertIn("zehn Uhr einundzwanzig", answer)
        self.assertEqual(self.ui.voice_payload["time"], "10:21")
        self.assertEqual(self.ui.voice_payload["days"], ["So"])

    def test_set_alarm_compound_minute_word(self):
        answer = self.router.handle("Stell jeden Tag einen Wecker um sechs Uhr fuenfundvierzig")
        self.assertEqual(answer, "Meinst du morgens oder abends?")

        answer = self.router.handle("morgens")

        self.assertIn("viertel vor sieben", answer)
        self.assertEqual(self.ui.voice_payload["time"], "06:45")

    def test_deactivate_all_alarms(self):
        answer = self.router.handle("Deaktiviere alle Wecker")
        self.assertIn("Soll ich alle Wecker deaktivieren?", answer)
        self.assertEqual(self.ui.voice_payload["type"], "set_all_active")
        self.assertFalse(self.ui.voice_payload["active"])

    def test_delete_alarm_by_time(self):
        answer = self.router.handle("L\u00f6sche den Wecker um sieben Uhr")
        self.assertIn("Soll ich Wecker 1 loeschen?", answer)
        self.assertEqual(self.ui.voice_payload["type"], "delete")
        self.assertEqual(self.ui.voice_payload["id"], "1")

    def test_pending_time_followup(self):
        first = self.router.handle("Stell einen Wecker")
        self.assertIn("Um wie viel Uhr", first)
        second = self.router.handle("um halb vier")
        self.assertEqual(second, "Meinst du morgens oder abends?")
        third = self.router.handle("morgens")
        self.assertEqual(third, "Jeden Tag?")
        fourth = self.router.handle("ja")
        self.assertIn("halb vier", fourth)
        self.assertEqual(self.ui.voice_payload["time"], "03:30")

    def test_voice_confirmation_yes(self):
        self.ui.current_frame = "voice_confirm"
        self.ui.voice_payload = {"type": "delete", "id": "1"}
        answer = self.router.handle("ja")
        self.assertIn("bestaetigt", answer)
        self.assertTrue(self.ui.executed_payload)

    def test_stop_alarm_voice_command_is_not_local_action(self):
        answer = self.router.handle("Alarm stoppen")

        self.assertIsNone(answer)
        self.assertFalse(self.ui.alarm_manager.stopped)

    def test_qwen_stop_alarm_action_is_rejected(self):
        assistant = FakeQwenAssistant(
            '{"action":"stop_alarm","hour":null,"minute":null,'
            '"days":null,"target_id":null,"delete_all":false}'
        )
        router = VoiceCommandRouter(self.ui, assistant)

        answer = router.handle("welcher alarm klingelt gerade")

        self.assertIsNone(answer)
        self.assertFalse(self.ui.alarm_manager.stopped)


if __name__ == "__main__":
    unittest.main()
