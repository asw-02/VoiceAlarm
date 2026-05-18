#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Handles alarm checking and buzzer/hardware triggering.
"""

import threading
import time
from datetime import datetime, timedelta

from config import Config
from hardware.sensors import Buzzer, HardwareButton, Solenoid


class AlarmManager:
    """Manages active alarms, rings the buzzer, and handles pre-alarms."""

    def __init__(self, ui_reference):
        self.ui = ui_reference
        self.buzzer = Buzzer(Config.PIN_BUZZER)
        self.solenoid = Solenoid(Config.PIN_MOSFET)
        self.btn_stop = HardwareButton(
            Config.PIN_AUTO,
            pull_up=True,
            bounce_time=Config.BOUNCE_TIME,
        )

        self.btn_stop.set_when_pressed(self.stop_alarm)
        self.last_pre_alarm_minute = ""
        self.stop_event = threading.Event()
        self.alarm_thread = None
        self.last_triggered_minute = ""

    def is_ringing(self) -> bool:
        return (
            self.alarm_thread is not None
            and self.alarm_thread.is_alive()
            and not self.stop_event.is_set()
        )

    @staticmethod
    def _alarm_matches(alarm, date_str, day_str, time_str):
        if not alarm.get("active") or alarm.get("time") != time_str:
            return False

        alarm_date = alarm.get("date")
        if alarm_date:
            return alarm_date == date_str

        return day_str in alarm.get("days", [])

    @staticmethod
    def _alarm_datetime(alarm):
        if not alarm.get("date") or not alarm.get("time"):
            return None

        try:
            return datetime.strptime(f"{alarm['date']} {alarm['time']}", "%Y-%m-%d %H:%M")
        except (TypeError, ValueError):
            return None

    def _refresh_alarm_views(self):
        if hasattr(self.ui, "refresh_wecker_list"):
            self.ui.refresh_wecker_list()

    def _delete_alarm(self, alarm_id):
        alarms = self.ui.db.data.get("wecker", {})
        if alarm_id not in alarms:
            return

        del alarms[alarm_id]
        self.ui.db.save()
        self._refresh_alarm_views()

    def _cleanup_expired_dated_alarms(self, now):
        current_minute = now.replace(second=0, microsecond=0)
        expired_ids = []

        for wid, alarm in list(self.ui.db.data.get("wecker", {}).items()):
            alarm_time = self._alarm_datetime(alarm)
            if alarm_time and alarm_time < current_minute:
                expired_ids.append(wid)

        if not expired_ids:
            return

        alarms = self.ui.db.data.get("wecker", {})
        for wid in expired_ids:
            alarms.pop(wid, None)

        self.ui.db.save()
        self._refresh_alarm_views()

    def check(self):
        now = self.ui.get_now()
        self._cleanup_expired_dated_alarms(now)

        now_str = now.strftime("%H:%M")
        date_str = now.strftime("%Y-%m-%d")
        day_str = Config.DAYS[now.weekday()]

        if now_str != self.last_triggered_minute:
            for wid, alarm in list(self.ui.db.data["wecker"].items()):
                if self._alarm_matches(alarm, date_str, day_str, now_str):
                    self.start_alarm(now_str)
                    if alarm.get("date"):
                        self._delete_alarm(wid)
                    break

        future_time = now + timedelta(seconds=30)
        future_str = future_time.strftime("%H:%M")
        future_date_str = future_time.strftime("%Y-%m-%d")
        future_day_str = Config.DAYS[future_time.weekday()]

        if now.second == 30 and future_str != self.last_pre_alarm_minute:
            for wid, alarm in list(self.ui.db.data["wecker"].items()):
                if self._alarm_matches(alarm, future_date_str, future_day_str, future_str):
                    self.last_pre_alarm_minute = future_str
                    self.solenoid.fire(duration_sec=2.0)
                    break

    def start_alarm(self, minute: str):
        if self.is_ringing():
            return

        self.last_triggered_minute = minute
        self.stop_event.clear()
        self.alarm_thread = threading.Thread(target=self._run_alarm_loop, daemon=True)
        self.alarm_thread.start()

    def _run_alarm_loop(self):
        tones = [(2048, 0.2), (2500, 0.2), (3000, 0.2)]
        while not self.stop_event.is_set():
            for freq, dur in tones:
                if self.stop_event.is_set():
                    break
                self.buzzer.play_tone(freq, dur)
                time.sleep(0.1)
        self.buzzer.off()

    def stop_alarm(self):
        self.stop_event.set()
        self.buzzer.off()
        print(">>> [Hardware] Alarm gestoppt.")

    def shutdown(self):
        self.stop_alarm()
        if self.alarm_thread and self.alarm_thread.is_alive():
            self.alarm_thread.join(timeout=1.0)
        self.buzzer.close()
        self.solenoid.close()
        self.btn_stop.close()
