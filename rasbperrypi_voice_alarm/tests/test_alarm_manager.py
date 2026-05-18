#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import sys
import types
import unittest
from datetime import datetime


class DummyGPIODevice:
    def __init__(self, *args, **kwargs):
        self.value = 0
        self.frequency = None
        self.when_pressed = None

    def on(self):
        pass

    def off(self):
        pass

    def close(self):
        pass


sys.modules.setdefault(
    "gpiozero",
    types.SimpleNamespace(
        DigitalInputDevice=DummyGPIODevice,
        Button=DummyGPIODevice,
        LED=DummyGPIODevice,
        PWMOutputDevice=DummyGPIODevice,
        OutputDevice=DummyGPIODevice,
    ),
)

from core.alarm_manager import AlarmManager


class FakeDB:
    def __init__(self, alarms):
        self.lock = threading.RLock()
        self.data = {"wecker": alarms}
        self.saved = False

    def save(self):
        self.saved = True


class FakeUI:
    def __init__(self, now, alarms):
        self._now = now
        self.db = FakeDB(alarms)
        self.refreshed = False

    def get_now(self):
        return self._now

    def refresh_wecker_list(self):
        self.refreshed = True


def make_manager(ui):
    manager = AlarmManager.__new__(AlarmManager)
    manager.ui = ui
    manager.last_pre_alarm_minute = ""
    manager.last_triggered_minute = ""
    manager.started_minute = None
    manager.solenoid = None
    manager.start_alarm = lambda minute: setattr(manager, "started_minute", minute)
    return manager


class AlarmManagerOneShotTests(unittest.TestCase):
    def test_dated_alarm_is_deleted_after_trigger(self):
        ui = FakeUI(
            datetime(2026, 5, 19, 3, 30),
            {
                "1": {
                    "time": "03:30",
                    "days": ["Di"],
                    "active": True,
                    "date": "2026-05-19",
                }
            },
        )
        manager = make_manager(ui)

        manager.check()

        self.assertEqual(manager.started_minute, "03:30")
        self.assertNotIn("1", ui.db.data["wecker"])
        self.assertTrue(ui.db.saved)
        self.assertTrue(ui.refreshed)

    def test_expired_dated_alarm_is_cleaned_up(self):
        ui = FakeUI(
            datetime(2026, 5, 19, 3, 31),
            {
                "1": {
                    "time": "03:30",
                    "days": ["Di"],
                    "active": True,
                    "date": "2026-05-19",
                }
            },
        )
        manager = make_manager(ui)

        manager.check()

        self.assertIsNone(manager.started_minute)
        self.assertNotIn("1", ui.db.data["wecker"])
        self.assertTrue(ui.db.saved)
        self.assertTrue(ui.refreshed)


if __name__ == "__main__":
    unittest.main()
