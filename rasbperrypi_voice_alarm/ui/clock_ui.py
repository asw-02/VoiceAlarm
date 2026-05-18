#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main UI renderer. Handles menus, drawing loops, and button actions.
"""

import time
import threading
import random
from datetime import datetime, timedelta
from config import Config

from core.alarm_manager import AlarmManager
from ui.manual_setup import ManualStartTime
from hardware.sensors import HardwareButton, LCDManager, RTCManager

class AlarmClockUI:
    def __init__(self, db, start_data, light_sensor=None):
        self.db = db
        self.light_sensor = light_sensor 
        self.apply_start_data(start_data)
        
        self.running = True
        self._stop_event = threading.Event()
        self.state_lock = threading.RLock()
        
        self.current_frame = "start"
        self.last_frame = None
        self.selected_index = 0
        self.edit_mode = None
        self.current_edit_id = None
        
        self.voice_payload = None
        self.voice_confirm_text = ("", "")
        self.voice_status_active = False
        self.voice_status_text = ("", "")
        self.voice_return_frame = "start"
        
        self.temp_msg_end = 0
        self.temp_msg_content = None

        self.time_digits = [0, 0, 0, 0]
        self.math_digits = [0, 0, 0, 0]
        self.digit_index = 0
        self.day_index = 0
        
        self.math_task = ""
        self.math_result = 0
        
        # --- Hardware Initialisierung ---
        self.btn_menu = HardwareButton(Config.PIN_MENU)
        self.btn_up = HardwareButton(Config.PIN_UP)
        self.btn_down = HardwareButton(Config.PIN_DOWN)
        self.btn_save = HardwareButton(Config.PIN_SAVE)
        
        self.lcd = LCDManager()
        self.rtc = RTCManager()
        
        self.rtc.sync_time(self.get_now())
        # --------------------------------
        
        self.menu_tree = {
            "start": ["Menue"],
            "menu": ["Wecker", "Systemeinstellungen", "Abschaltung"],
            "wecker": [],
            "system": ["Masterrechnung", "Zeit/Datum aendern", "Werkseinstellungen laden"],
            "confirm_reset": ["Abbrechen", "JA, ALLES LOESCHEN"],
            "abschaltung": ["Manuell abschalten"],
            "confirm_exit": ["Abbrechen", "JA, ABSCHALTEN"]
        }
        self.refresh_wecker_list()
        self.alarm_manager = AlarmManager(self)
        
        self.BIG_FONT_CHARS = [
            [0x07, 0x0F, 0x1F, 0x1F, 0x1F, 0x1F, 0x1F, 0x1F],
            [0x1C, 0x1E, 0x1F, 0x1F, 0x1F, 0x1F, 0x1F, 0x1F],
            [0x1F, 0x1F, 0x1F, 0x1F, 0x1F, 0x1F, 0x0F, 0x07],
            [0x1F, 0x1F, 0x1F, 0x1F, 0x1F, 0x1F, 0x1E, 0x1C],
            [0x1F, 0x1F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F, 0x1F],
            [0x1F, 0x1F, 0x00, 0x00, 0x00, 0x00, 0x1F, 0x1F],
            [0x00, 0x0E, 0x1F, 0x1F, 0x1F, 0x0E, 0x00, 0x00],
        ]

        self.DIGIT_MAP = {
            "0": [(0, 4, 1), (2, 5, 3)], "1": [(1,), (3,)],
            "2": [(6, 6, 1), (2, 6, 6)], "3": [(4, 6, 1), (5, 6, 3)],
            "4": [(0, 5, 1), (32, 32, 3)], "5": [(0, 6, 6), (6, 6, 3)],
            "6": [(0, 6, 6), (2, 6, 3)], "7": [(4, 4, 1), (32, 2, 32)],
            "8": [(0, 6, 1), (2, 6, 3)], "9": [(0, 6, 1), (6, 6, 3)],
            " ": [(32, 32, 32), (32, 32, 32)], ":": [(7,), (7,)],
        }
        
        self.load_big_font()
        
        self.marquee_offset = 0
        self.last_marquee_time = 0
        self.last_marquee_line = ""
        self.marquee_speed = 0.35
        
        self.alarm_active = False
        self.button_pressed = False
        self.last_button_time = 0

    def stop(self):
        self.running = False
        self._stop_event.set()

    def generate_math_task(self):
        val1 = random.randint(11, 99)
        val2 = random.randint(11, 99)
        self.math_task = f"{val1} x {val2} ="
        self.math_result = val1 * val2
        self.math_digits = [0, 0, 0, 0]
        self.digit_index = 0

    def apply_start_data(self, data):
        d, m, y = map(int, data["date"].split('.'))
        hh, mm, ss = map(int, data["time"].split(':'))
        self.base_datetime = datetime(y, m, d, hh, mm, ss)
        self.start_ts = data["start_ts"]

    def get_now(self):
        elapsed = time.time() - self.start_ts
        return self.base_datetime + timedelta(seconds=elapsed)

    def show_temp_message(self, line1, line2, duration=3):
        l1 = str(line1).center(Config.LCD_COLS) if line1 else " " * Config.LCD_COLS
        l2 = str(line2).center(Config.LCD_COLS) if line2 else " " * Config.LCD_COLS
        
        with self.state_lock: 
            self.temp_msg_content = (l1, l2)
            self.temp_msg_end = time.time() + duration
        self.draw(force=True)

    def request_confirmation(self, line1, line2, payload):
        with self.state_lock:
            self.voice_confirm_text = (line1, line2)
            self.voice_payload = payload
            self.current_frame = "voice_confirm"
        self.draw(force=True)

    def set_voice_status(self, line1, line2=""):
        with self.state_lock:
            self.temp_msg_content = None
            self.temp_msg_end = 0

            if self.current_frame not in {"voice_status", "voice_confirm", "overwrite_select"}:
                self.voice_return_frame = self.current_frame
                self.current_frame = "voice_status"
            elif self.current_frame == "voice_status":
                self.current_frame = "voice_status"

            self.voice_status_active = True
            self.voice_status_text = (str(line1 or ""), str(line2 or ""))
        self.draw(force=True)

    def end_voice_status(self):
        with self.state_lock:
            self.voice_status_active = False
            self.voice_status_text = ("", "")
            if self.current_frame == "voice_status":
                self.current_frame = self.voice_return_frame or "start"
                self.voice_return_frame = "start"
        self.draw(force=True)

    def execute_voice_payload(self):
        if not self.voice_payload: return
        
        p_type = self.voice_payload.get("type")
        message_line1 = None
        message_line2 = None
        next_frame = "start" 
        
        try:
            with self.db.lock:
                if p_type == "create":
                    existing_ids = []
                    for k in self.db.data["wecker"].keys():
                        try: existing_ids.append(int(k))
                        except ValueError: continue

                    new_id = None
                    for i in range(1, Config.MAX_WECKER + 1):
                        if i not in existing_ids:
                            new_id = i
                            break
                    
                    if new_id is not None:
                        alarm_data = {
                            "time": self.voice_payload["time"],
                            "days": self.voice_payload["days"],
                            "active": True
                        }
                        if self.voice_payload.get("date"):
                            alarm_data["date"] = self.voice_payload["date"]

                        self.db.data["wecker"][str(new_id)] = alarm_data
                        self.db.save()
                        message_line1, message_line2 = "Gespeichert!", f"Auf Platz {new_id}"
                    else:
                        self.overwrite_mapping = sorted(list(self.db.data["wecker"].keys()), key=int)
                        self.voice_payload["type"] = "overwrite_menu"
                        self.selected_index = 0
                        next_frame = "overwrite_select"

                elif p_type == "delete":
                    wid = str(self.voice_payload.get("id"))
                    if wid in self.db.data["wecker"]:
                        del self.db.data["wecker"][wid]
                        self.db.save()
                        message_line1, message_line2 = "Geloescht!", f"Wecker {wid} geloescht"
                    else:
                        message_line1, message_line2 = "Fehler", "ID nicht gefunden"

                elif p_type == "delete_all":
                    self.db.data["wecker"] = {}
                    self.db.save()
                    message_line1, message_line2 = "Alles leer!", "Datenbank bereinigt"

                elif p_type == "set_active":
                    wid = str(self.voice_payload.get("id"))
                    active = bool(self.voice_payload.get("active"))
                    if wid in self.db.data["wecker"]:
                        self.db.data["wecker"][wid]["active"] = active
                        self.db.save()
                        status = "aktiviert" if active else "deaktiviert"
                        message_line1, message_line2 = "Gespeichert!", f"W{wid} {status}"
                    else:
                        message_line1, message_line2 = "Fehler", "ID nicht gefunden"

                elif p_type == "set_all_active":
                    active = bool(self.voice_payload.get("active"))
                    for alarm in self.db.data["wecker"].values():
                        alarm["active"] = active
                    self.db.save()
                    status = "aktiviert" if active else "deaktiviert"
                    message_line1, message_line2 = "Gespeichert!", f"Alle {status}"

        except Exception as e:
            self.show_temp_message("SYSTEM FEHLER", "Siehe Konsole")
            self.voice_payload = None
            self.current_frame = "start"
            return

        self.refresh_wecker_list()
        if message_line1: self.show_temp_message(message_line1, message_line2)
        
        if next_frame == "start":
            self.voice_payload = None
            self.current_frame = "start"
        else:
            self.current_frame = next_frame
            self.last_button_time = time.time() + 1.0 
        self.draw(force=True)

    def refresh_wecker_list(self):
        w_list = sorted(self.db.data["wecker"].keys(), key=int)
        self.menu_tree["wecker"] = [f"Wecker {wid}" for wid in w_list]
        if len(w_list) < Config.MAX_WECKER: self.menu_tree["wecker"].append("Wecker hinzufuegen")
        for wid in w_list:
            w = self.db.data["wecker"][wid]
            stat = "An" if w["active"] else "Aus"
            self.menu_tree[f"wecker{wid}"] = ["Uhrzeit einstellen", "Tage auswaehlen", f"Status: {stat}", "Loeschen"]

    def get_next_alarm_info(self):
        now = self.get_now(); best_dt, best_id = None, None
        for wid, w in self.db.data["wecker"].items():
            if not w["active"]: continue
            hh, mm = map(int, w["time"].split(":"))
            if w.get("date"):
                try:
                    cand = datetime.strptime(f"{w['date']} {w['time']}", "%Y-%m-%d %H:%M")
                except ValueError:
                    continue
                if cand <= now:
                    continue
                if best_dt is None or cand < best_dt: best_dt, best_id = cand, wid
                continue

            if not w["days"]: continue
            for d in w["days"]:
                d_idx = Config.DAYS.index(d)
                diff = (d_idx - now.weekday()) % 7
                cand = now.replace(hour=hh, minute=mm, second=0, microsecond=0) + timedelta(days=diff)
                if cand <= now: cand += timedelta(days=7)
                if best_dt is None or cand < best_dt: best_dt, best_id = cand, wid
        if best_id:
            w = self.db.data["wecker"][best_id]
            return f"W{best_id}", w["time"], ",".join(w["days"])
        return "Kein Wecker", "", ""

    def update_backlight(self):
        new_state = True 
        if self.light_sensor and self.light_sensor.is_dark():
            is_ringing = self.alarm_manager.is_ringing()
            recent_button_press = (time.time() - self.last_button_time) <= 5.0
            if not is_ringing and not recent_button_press:
                new_state = False
        self.lcd.set_backlight(new_state)
    
    def center_text(self, text):
        if len(text) >= Config.LCD_COLS: return text[:Config.LCD_COLS]
        pad = (Config.LCD_COLS - len(text)) // 2
        return " " * pad + text + " " * (Config.LCD_COLS - pad - len(text))

    def get_active_alarms_text(self):
        parts = []
        for wid, w in sorted(self.db.data["wecker"].items(), key=lambda x: int(x[0])):
            if not w["active"]: continue
            if w.get("date"):
                try:
                    alarm_date = datetime.strptime(w["date"], "%Y-%m-%d")
                    days = alarm_date.strftime("%d.%m.")
                except ValueError:
                    days = w["date"]
            else:
                days = ",".join(w["days"])
            parts.append(f"W{wid} {w['time']} ({days})")
        return " | ".join(parts) if parts else "Keine aktiven Wecker"

    def marquee(self, text):
        if not text: return " " * 20
        if self.alarm_active: return self.center_text(text)
        if self.button_pressed or (time.time() - self.last_marquee_time < self.marquee_speed):
            return self.last_marquee_line

        self.last_marquee_time = time.time()
        padded = text + "   "
        self.marquee_offset = (self.marquee_offset + 1) % len(padded)
        line = padded[self.marquee_offset:self.marquee_offset + 20].ljust(20)
        self.last_marquee_line = line
        return line

    def load_big_font(self):
        for i, bitmap in enumerate(self.BIG_FONT_CHARS):
            self.lcd.create_char(i, bitmap)
            
    def draw_big_time(self, time_str, row=0, col=None):
        if hasattr(self, "_last_big_time") and self._last_big_time == time_str: return 
        self._last_big_time = time_str 

        self.lcd.write_custom_chars(row, 0, [32] * Config.LCD_COLS)
        self.lcd.write_custom_chars(row + 1, 0, [32] * Config.LCD_COLS)

        widths = []
        for index, ch in enumerate(time_str):
            top, _ = self.DIGIT_MAP.get(ch, self.DIGIT_MAP[" "])
            width = len(top)

            next_ch = time_str[index + 1] if index + 1 < len(time_str) else None
            if next_ch and ch != ":" and next_ch != ":":
                width += 1

            widths.append(width)

        total_width = sum(widths)
        if col is None:
            col = max(0, (Config.LCD_COLS - total_width) // 2)

        positions = []
        cursor = col
        for index, ch in enumerate(time_str):
            positions.append((cursor, ch))
            cursor += widths[index]
        
        # Oben
        for cursor, ch in positions:
            top, _ = self.DIGIT_MAP.get(ch, self.DIGIT_MAP[" "])
            self.lcd.write_custom_chars(row, cursor, top)

        # Unten
        for cursor, ch in positions:
            _, bottom = self.DIGIT_MAP.get(ch, self.DIGIT_MAP[" "])
            self.lcd.write_custom_chars(row + 1, cursor, bottom)

    def draw(self, force=False):
        if not self.running: return
        self.update_backlight()
        
        if time.time() < self.temp_msg_end and self.temp_msg_content:
            self.lcd.write_line(0, self.center_text(" *** INFO ***"))
            self.lcd.write_line(1, self.temp_msg_content[0])
            self.lcd.write_line(2, self.temp_msg_content[1])
            self.lcd.write_line(3, "")
            return
            
        now = self.get_now()
        if self.last_frame != self.current_frame or force:
            self.lcd.clear()
            self.last_frame = self.current_frame
        
        if self.edit_mode == "math_input":
            self.lcd.write_line(0, "Masterrechnung:")
            self.lcd.write_line(1, f"{self.math_task} {''.join(map(str, self.math_digits))}")
            self.lcd.write_line(2, " " * (len(self.math_task) + 1 + self.digit_index) + "^")
        
        elif self.edit_mode == "time_input":
            hh, mm = f"{self.time_digits[0]}{self.time_digits[1]}", f"{self.time_digits[2]}{self.time_digits[3]}"
            self.lcd.write_line(0, "Zeit einstellen:"); self.lcd.write_line(1, f"      {hh}:{mm}")
            self.lcd.write_line(2, " " * [6, 7, 9, 10][self.digit_index] + "^")
        
        elif self.current_frame == "tage":
            wid = str(self.current_edit_id); sel = self.db.data["wecker"][wid]["days"]
            opts = [f"{('[X]' if d in sel else '[ ]')} {d}" for d in Config.DAYS] + ["Speichern"]
            start = max(0, min(self.day_index - 1, len(opts) - 4))
            for i in range(min(4, len(opts))):
                curr = start + i
                self.lcd.write_line(i, f"{('>' if curr == self.day_index else ' ')}{opts[curr]}")
        
        elif self.current_frame == "voice_confirm":
            self.lcd.write_line(0, "Sprachbefehl erkannt:")
            self.lcd.write_line(1, self.voice_confirm_text[0])
            self.lcd.write_line(2, self.voice_confirm_text[1])
            self.lcd.write_line(3, "Save=JA   Menu=NEIN")
            return

        elif self.current_frame == "voice_status":
            line1, line2 = self.voice_status_text
            self.lcd.write_line(0, self.center_text("Sprachsteuerung"))
            self.lcd.write_line(1, self.center_text(line1))
            self.lcd.write_line(2, self.center_text(line2))
            self.lcd.write_line(3, self.center_text("aktiv"))
            return
        
        elif self.current_frame == "start":
            time_str = now.strftime("%H:%M")
            self.draw_big_time(time_str, row=0)
            date_str = f"{Config.DAYS[now.weekday()]} {now.strftime('%d.%m.%Y')}"
            self.lcd.write_line(2, self.center_text(date_str))
            alarms_text = self.get_active_alarms_text()
            self.lcd.write_line(3, self.marquee(alarms_text))
            return
            
        elif self.current_frame == "overwrite_select":
            self.lcd.write_line(0, "Welchen ersetzen?")
            start = max(0, min(self.selected_index - 1, len(self.overwrite_mapping) - 3))
            
            for i in range(min(3, len(self.overwrite_mapping))):
                curr = start + i
                if curr >= len(self.overwrite_mapping): break
                
                wid = str(self.overwrite_mapping[curr])
                w_data = self.db.data["wecker"][wid]
                
                alte_zeit = w_data.get("time", "--:--")
                ist_aktiv = w_data.get("active", False)
                status_text = "An " if ist_aktiv else "Aus"
                
                d_list = w_data.get("days", [])
                if len(d_list) == 7: d_str = "Mo-So"
                elif d_list == ["Sa", "So"]: d_str = "Sa,So"
                elif d_list == ["Mo", "Di", "Mi", "Do", "Fr"]: d_str = "Mo-Fr"
                elif not d_list: d_str = "Einmalig"
                else: d_str = ",".join(d_list)
                
                day_space = 6
                if len(d_str) > day_space:
                    padded = d_str + "   " 
                    doubled = padded + padded 
                    offset = int(time.time() / 0.4) % len(padded)
                    if curr == self.selected_index: disp_days = doubled[offset : offset + day_space]
                    else: disp_days = d_str[:day_space] 
                else:
                    disp_days = d_str[:day_space].ljust(day_space)
                    
                prefix = ">" if curr == self.selected_index else " "
                line = f"{prefix}W{wid} {alte_zeit} {disp_days} {status_text}"
                self.lcd.write_line(i + 1, line)
        else:
            opts = self.menu_tree.get(self.current_frame, [])
            start = max(0, min(self.selected_index - 1, len(opts) - 4))
            for i in range(min(4, len(opts))):
                curr = start + i
                self.lcd.write_line(i, f"{('>' if curr == self.selected_index else ' ')}{opts[curr]}")
   
    def check_buttons(self):
        pressed = (self.btn_up.is_pressed or self.btn_down.is_pressed or 
                   self.btn_menu.is_pressed or self.btn_save.is_pressed)

        self.button_pressed = pressed
        if not pressed:
            self._button_handled = False 
            return

        if getattr(self, "_button_handled", False): return 

        self._button_handled = True
        self.last_button_time = time.time()

        if self.btn_up.is_pressed:
            if self.edit_mode == "math_input":
                self.math_digits[self.digit_index] = (self.math_digits[self.digit_index] + 1) % 10
            elif self.edit_mode == "time_input":
                lim = [2, (3 if self.time_digits[0] == 2 else 9), 5, 9][self.digit_index]
                self.time_digits[self.digit_index] = (self.time_digits[self.digit_index] + 1) % (lim + 1)
            elif self.current_frame == "tage":
                self.day_index = (self.day_index - 1) % 8
            elif self.current_frame == "overwrite_select":
                if hasattr(self, "overwrite_mapping") and self.overwrite_mapping:
                    self.selected_index = (self.selected_index - 1) % len(self.overwrite_mapping)
            else:
                opts = self.menu_tree.get(self.current_frame, [])
                if opts: self.selected_index = (self.selected_index - 1) % len(opts)

            self.draw(force=True)
            return

        if self.btn_down.is_pressed:
            if self.edit_mode == "math_input":
                self.math_digits[self.digit_index] = (self.math_digits[self.digit_index] - 1) % 10
            elif self.edit_mode == "time_input":
                lim = [2, (3 if self.time_digits[0] == 2 else 9), 5, 9][self.digit_index]
                self.time_digits[self.digit_index] = (self.time_digits[self.digit_index] - 1) % (lim + 1)
            elif self.current_frame == "tage":
                self.day_index = (self.day_index + 1) % 8
            elif self.current_frame == "overwrite_select":
                if hasattr(self, "overwrite_mapping") and self.overwrite_mapping:
                    self.selected_index = (self.selected_index + 1) % len(self.overwrite_mapping)
            else:
                opts = self.menu_tree.get(self.current_frame, [])
                if opts: self.selected_index = (self.selected_index + 1) % len(opts)

            self.draw(force=True)
            return

        if self.btn_menu.is_pressed:
            if self.current_frame in ["voice_confirm", "overwrite_select"]:
                self.current_frame = "start"
                self.voice_payload = None
                self.show_temp_message("Abgebrochen", "")
                self.draw(force=True)
                return
            
            if self.edit_mode:
                self.edit_mode = None
                self.current_frame = "system"
            elif self.current_frame == "tage":
                self.current_frame = f"wecker{self.current_edit_id}"
            elif self.current_frame in ["menu", "start"]:
                self.current_frame = "start"
            else:
                self.current_frame = "menu"

            self.selected_index = 0
            self.draw(force=True)
            return

        if self.btn_save.is_pressed:
            self.handle_select()
            self.draw(force=True)
            return

    def handle_select(self):
        if self.current_frame == "voice_confirm":
            self.execute_voice_payload()
            return
        
        if self.current_frame == "overwrite_select":
            wid = str(self.overwrite_mapping[self.selected_index])
            alarm_data = {
                "time": self.voice_payload["time"],
                "days": self.voice_payload["days"],
                "active": True
            }
            if self.voice_payload.get("date"):
                alarm_data["date"] = self.voice_payload["date"]

            self.db.data["wecker"][wid] = alarm_data
            self.db.save()
            self.show_temp_message("Ueberschrieben", f"Wecker {wid} ersetzt")
            self.current_frame = "start"
            self.voice_payload = None
            self.refresh_wecker_list()
            self.draw(force=True)
            return
        
        if self.edit_mode == "math_input":
            self.digit_index += 1
            if self.digit_index > 3:
                user_res = int("".join(map(str, self.math_digits)))
                if user_res == self.math_result:
                    self.alarm_manager.stop_alarm()
                    self.edit_mode = None; self.current_frame = "start"
                else:
                    self.lcd.write_line(3, "FALSCH! Neu...")
                    time.sleep(1)
                    self.generate_math_task()
            self.draw(True) 
            return

        if self.edit_mode == "time_input":
            self.digit_index += 1
            if self.digit_index > 3:
                new_t = f"{self.time_digits[0]}{self.time_digits[1]}:{self.time_digits[2]}{self.time_digits[3]}"
                self.db.data["wecker"][str(self.current_edit_id)]["time"] = new_t
                self.db.save(); self.edit_mode = None; self.current_frame = f"wecker{self.current_edit_id}"
            self.draw(True); return

        if self.current_frame == "tage":
            if self.day_index < 7:
                d, ds = Config.DAYS[self.day_index], self.db.data["wecker"][str(self.current_edit_id)]["days"]
                if d in ds: ds.remove(d)
                else: ds.append(d); ds.sort(key=lambda x: Config.DAYS.index(x))
                self.db.save(); self.draw(True)
            else: self.current_frame = f"wecker{self.current_edit_id}"; self.day_index = 0; self.draw(True)
            return

        opts = self.menu_tree.get(self.current_frame, [])
        if not opts: return
        sel = opts[self.selected_index]

        if sel == "Masterrechnung":
            if self.alarm_manager.is_ringing():
                self.generate_math_task(); self.edit_mode = "math_input"
            else:
                self.lcd.write_line(2, "Kein aktiver Alarm")
                time.sleep(1.5)
        elif sel == "JA, ABSCHALTEN":
            if self.alarm_manager.is_ringing():
                self.lcd.write_line(3, "Alarm aktiv! Sperre")
                time.sleep(1.5); self.draw(True)
            else: self.stop()
        elif sel == "Wecker": self.current_frame = "wecker"
        elif sel == "Systemeinstellungen": self.current_frame = "system"
        elif sel == "Abschaltung": self.current_frame = "abschaltung"
        elif sel == "Manuell abschalten": self.current_frame = "confirm_exit"
        elif sel == "Abbrechen": self.current_frame = "menu"
        elif sel == "Zeit/Datum aendern":
            self.btn_menu.close(); self.btn_up.close(); self.btn_down.close(); self.btn_save.close()
            m_input = ManualStartTime(self.lcd)
            new_data = m_input.run(); self.apply_start_data(new_data)
            
            # Restart Hardware
            self.btn_menu = HardwareButton(Config.PIN_MENU)
            self.btn_up = HardwareButton(Config.PIN_UP)
            self.btn_down = HardwareButton(Config.PIN_DOWN)
            self.btn_save = HardwareButton(Config.PIN_SAVE)
            
            self.rtc.sync_time(self.get_now()); self.current_frame = "start"
        elif sel == "Werkseinstellungen laden": self.current_frame = "confirm_reset"
        elif sel == "JA, ALLES LOESCHEN":
            self.db.data["wecker"] = {}; self.db.save(); self.refresh_wecker_list(); self.current_frame = "menu"
        elif sel == "Wecker hinzufuegen":
            nid = 1
            while str(nid) in self.db.data["wecker"]: nid += 1
            self.db.data["wecker"][str(nid)] = {"time": "00:00", "days": [], "active": False}
            self.db.save(); self.refresh_wecker_list()
        elif sel.startswith("Wecker "): self.current_edit_id, self.current_frame = sel.split()[-1], f"wecker{sel.split()[-1]}"
        elif sel == "Uhrzeit einstellen":
            self.edit_mode, self.digit_index = "time_input", 0
            self.time_digits = [int(d) for d in self.db.data["wecker"][str(self.current_edit_id)]["time"].replace(":", "")]
        elif sel == "Tage auswaehlen": self.current_frame = "tage"
        elif sel.startswith("Status:"):
            w = self.db.data["wecker"][str(self.current_edit_id)]
            w["active"] = not w["active"]; self.db.save(); self.refresh_wecker_list()
        elif sel == "Loeschen":
            del self.db.data["wecker"][str(self.current_edit_id)]; self.db.save(); self.refresh_wecker_list(); self.current_frame = "wecker"
        elif self.current_frame == "start": self.current_frame = "menu"
        
        self.selected_index = 0; self.draw(True)

    def shutdown_hw(self):
        self.running = False
        self._stop_event.set()

        for button_name in ("btn_menu", "btn_up", "btn_down", "btn_save"):
            button = getattr(self, button_name, None)
            if button:
                button.close()

        if self.alarm_manager:
            self.alarm_manager.shutdown()

        if self.light_sensor and hasattr(self.light_sensor, "close"):
            self.light_sensor.close()

        if self.rtc and hasattr(self.rtc, "close"):
            self.rtc.close()

        if self.lcd:
            self.lcd.close()
