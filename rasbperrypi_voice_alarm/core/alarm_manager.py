#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Handles alarm checking and buzzer/hardware triggering.
"""

import time
import threading
from datetime import timedelta
from config import Config
from hardware.sensors import HardwareButton, Buzzer, Solenoid

class AlarmManager:
    """Manages active alarms, rings the buzzer, and handles pre-alarms."""
    
    def __init__(self, ui_reference):
        self.ui = ui_reference
        
        # Komplett abstrahiert
        self.buzzer = Buzzer(Config.PIN_BUZZER)
        self.solenoid = Solenoid(Config.PIN_MOSFET)
        self.btn_stop = HardwareButton(Config.PIN_AUTO, pull_up=True, bounce_time=Config.BOUNCE_TIME)
        
        self.btn_stop.set_when_pressed(self.stop_alarm) 
        self.last_pre_alarm_minute = "" 
        self.stop_event = threading.Event()
        self.alarm_thread = None
        self.last_triggered_minute = ""

    def is_ringing(self) -> bool:
        return self.alarm_thread is not None and self.alarm_thread.is_alive() and not self.stop_event.is_set()

    def check(self):
        now = self.ui.get_now()
        now_str = now.strftime("%H:%M")
        day_str = Config.DAYS[now.weekday()]

        if now_str != self.last_triggered_minute:
            for wid, w in self.ui.db.data["wecker"].items():
                if w["active"] and (day_str in w["days"]) and (w["time"] == now_str):
                    self.start_alarm(now_str)
                    break

        future_time = now + timedelta(seconds=30)
        future_str = future_time.strftime("%H:%M")
        future_day_str = Config.DAYS[future_time.weekday()]

        if now.second == 30 and future_str != self.last_pre_alarm_minute:
            for wid, w in self.ui.db.data["wecker"].items():
                if w["active"] and (future_day_str in w["days"]) and (w["time"] == future_str):
                    self.last_pre_alarm_minute = future_str
                    # self.solenoid.fire() # Aktiviert Hubmagnet für 1.5s
                    break

    def start_alarm(self, minute: str):
        if self.is_ringing(): return
        self.last_triggered_minute = minute
        self.stop_event.clear()
        if hasattr(self.ui, "speak"):
            self.ui.speak("Alarm.", interrupt=True)
        self.alarm_thread = threading.Thread(target=self._run_alarm_loop, daemon=True)
        self.alarm_thread.start()

    def _run_alarm_loop(self):
        tones = [(2048, 0.2), (2500, 0.2), (3000, 0.2)]
        while not self.stop_event.is_set():
            for freq, dur in tones:
                if self.stop_event.is_set(): break
                self.buzzer.play_tone(freq, dur)
                time.sleep(0.1)
        self.buzzer.off()

    def stop_alarm(self):
        self.stop_event.set()
        if hasattr(self.ui, "speak"):
            self.ui.speak("Alarm gestoppt.", interrupt=True)
        print(">>> [Hardware] Alarm gestoppt.")
