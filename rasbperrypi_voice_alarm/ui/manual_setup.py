#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Initial manual setup screen to configure the RTC / system time.
"""

import time
from datetime import datetime
from config import Config
from hardware.sensors import HardwareButton, LCDManager

class ManualStartTime:
    def __init__(self, lcd_manager: LCDManager = None):
        # Nutzt den übergebenen Manager oder baut sich zur Not einen eigenen
        self.lcd = lcd_manager if lcd_manager else LCDManager()
        
        self.btn_menu = HardwareButton(Config.PIN_MENU)
        self.btn_up = HardwareButton(Config.PIN_UP)
        self.btn_down = HardwareButton(Config.PIN_DOWN)
        self.btn_save = HardwareButton(Config.PIN_SAVE)

        self.date_digits = [0, 1, 0, 1, 2, 0, 2, 6] 
        self.time_digits = [0, 0, 0, 0, 0, 0]
        self.weekday_idx = 0
        self.phase = 0
        self.digit_ptr = 0
        self.finished = False

    def update_display(self):
        if self.phase == 0:
            self.lcd.write_line(0, "Datum eingeben:")
            d = self.date_digits
            self.lcd.write_line(1, f"  {d[0]}{d[1]}.{d[2]}{d[3]}.{d[4]}{d[5]}{d[6]}{d[7]}")
            pos = [2, 3, 5, 6, 8, 9, 10, 11][self.digit_ptr]
            self.lcd.write_line(2, " " * pos + "^")
        elif self.phase == 2:
            self.lcd.write_line(0, "Uhrzeit eingeben:")
            t = self.time_digits
            self.lcd.write_line(1, f"  {t[0]}{t[1]}:{t[2]}{t[3]}:{t[4]}{t[5]}")
            pos = [2, 3, 5, 6, 8, 9][self.digit_ptr]
            self.lcd.write_line(2, " " * pos + "^")
        self.lcd.write_line(3, "(Menu=Zurueck)")

    def run(self) -> dict:
        self.update_display()
        while not self.finished:
            if self.btn_up.is_pressed:
                if self.phase == 0:
                    lim = [3, 9, 1, 9, 9, 9, 9, 9][self.digit_ptr]
                    self.date_digits[self.digit_ptr] = (self.date_digits[self.digit_ptr] + 1) % (lim + 1)
                elif self.phase == 1: self.weekday_idx = (self.weekday_idx + 1) % 7
                elif self.phase == 2:
                    lim = [2, (3 if self.time_digits[0]==2 else 9), 5, 9, 5, 9][self.digit_ptr]
                    self.time_digits[self.digit_ptr] = (self.time_digits[self.digit_ptr] + 1) % (lim + 1)
                self.update_display(); time.sleep(0.2)
            elif self.btn_down.is_pressed:
                if self.phase == 0:
                    lim = [3, 9, 1, 9, 9, 9, 9, 9][self.digit_ptr]
                    self.date_digits[self.digit_ptr] = (self.date_digits[self.digit_ptr] - 1) % (lim + 1)
                elif self.phase == 1: self.weekday_idx = (self.weekday_idx - 1) % 7
                elif self.phase == 2:
                    lim = [2, (3 if self.time_digits[0]==2 else 9), 5, 9, 5, 9][self.digit_ptr]
                    self.time_digits[self.digit_ptr] = (self.time_digits[self.digit_ptr] - 1) % (lim + 1)
                self.update_display(); time.sleep(0.2)
            elif self.btn_save.is_pressed:
                if self.phase == 0:
                    if self.digit_ptr < 7: self.digit_ptr += 1
                    else: self.phase = 2; self.digit_ptr = 0 
                elif self.phase == 2:
                    if self.digit_ptr < 5: self.digit_ptr += 1
                    else: self.finished = True
                if not self.finished: self.update_display()
                time.sleep(0.3)
            elif self.btn_menu.is_pressed:
                if self.phase == 0 and self.digit_ptr > 0: self.digit_ptr -= 1
                elif self.phase == 2:
                    if self.digit_ptr > 0: self.digit_ptr -= 1
                    else: self.phase = 0; self.digit_ptr = 7 
                self.update_display(); time.sleep(0.3)
            time.sleep(0.05)
        
        d = self.date_digits
        t = self.time_digits
        try:
            day = int(f"{d[0]}{d[1]}")
            month = int(f"{d[2]}{d[3]}")
            year = int(f"{d[4]}{d[5]}{d[6]}{d[7]}")
            auto_weekday = datetime(year, month, day).weekday()
        except Exception:
            auto_weekday = 0 

        res = {
            "date": f"{d[0]}{d[1]}.{d[2]}{d[3]}.{d[4]}{d[5]}{d[6]}{d[7]}", 
            "weekday_idx": auto_weekday,
            "time": f"{t[0]}{t[1]}:{t[2]}{t[3]}:{t[4]}{t[5]}", 
            "start_ts": time.time()
        }
        self.btn_menu.close(); self.btn_up.close(); self.btn_down.close(); self.btn_save.close()
        return res
