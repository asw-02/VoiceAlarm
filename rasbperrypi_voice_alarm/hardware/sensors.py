#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wrapper for all hardware components (Sensors, Buttons, LEDs, Buzzer, Solenoid, LCD, RTC).
Centralizes hardware dependencies to isolate the main logic.
"""

import time
import threading
from gpiozero import DigitalInputDevice, Button, LED, PWMOutputDevice, OutputDevice
from config import Config, USE_LCD, SMBUS_AVAILABLE

if USE_LCD:
    try:
        from RPLCD.i2c import CharLCD
    except ImportError:
        pass

if SMBUS_AVAILABLE:
    try:
        import smbus
    except ImportError:
        smbus = None

# --- BASIC SENSORS & ACTUATORS ---

class LightSensor:
    def __init__(self, pin=Config.PIN_LIGHT_SENSOR, enabled=True):
        self.sensor = DigitalInputDevice(pin, pull_up=True)
        self.enabled = enabled  

    def is_dark(self) -> bool:
        if not self.enabled or not self.sensor: return False
        return not self.sensor.is_active

    def close(self):
        if self.sensor:
            self.sensor.close()
            self.sensor = None

class HardwareButton:
    def __init__(self, pin, pull_up=True, bounce_time=Config.BOUNCE_TIME):
        self._button = Button(pin, pull_up=pull_up, bounce_time=bounce_time)

    @property
    def is_pressed(self) -> bool: return bool(self._button and self._button.is_pressed)

    def set_when_pressed(self, callback_function):
        if self._button:
            self._button.when_pressed = callback_function

    def close(self):
        if self._button:
            self._button.close()
            self._button = None

class StatusLED:
    def __init__(self, pin): self._led = LED(pin)
    def on(self):
        if self._led: self._led.on()
    def off(self):
        if self._led: self._led.off()
    def close(self):
        if self._led:
            self._led.off()
            self._led.close()
            self._led = None

class Buzzer:
    def __init__(self, pin=Config.PIN_BUZZER):
        self._buzzer = PWMOutputDevice(pin)
        
    def play_tone(self, frequency, duration=None, volume=0.5):
        if not self._buzzer:
            return
        self._buzzer.frequency = frequency
        self._buzzer.value = volume
        if duration:
            time.sleep(duration)
            self._buzzer.value = 0
            
    def off(self):
        if self._buzzer:
            self._buzzer.off()

    def close(self):
        if self._buzzer:
            self._buzzer.off()
            self._buzzer.close()
            self._buzzer = None

class Solenoid:
    def __init__(self, pin=Config.PIN_MOSFET):
        self._solenoid = OutputDevice(pin)
        self._off_timer = None
        
    def activate_solenoid(self, duration_sec=2.0):
        if not self._solenoid:
            return

        print(">>> [Hardware] Hubmagnet AKTIVIERT (30s Pre-Alarm)")
        print("Hubmagnet AN (Stift sollte EINZIEHEN)")
        self._solenoid.on()

        if self._off_timer:
            self._off_timer.cancel()

        self._off_timer = threading.Timer(duration_sec, self._deactivate_solenoid)
        self._off_timer.daemon = True
        self._off_timer.start()

    def _deactivate_solenoid(self):
        if not self._solenoid:
            return

        print("Hubmagnet AUS (Stift sollte AUSFAHREN)")
        self._solenoid.off()
        self._off_timer = None

    def fire(self, duration_sec=2.0):
        self.activate_solenoid(duration_sec=duration_sec)

    def close(self):
        if self._off_timer:
            self._off_timer.cancel()
            self._off_timer = None
        if self._solenoid:
            self._solenoid.off()
            self._solenoid.close()
            self._solenoid = None

# --- COMPLEX MANAGERS (I2C) ---

class LCDManager:
    """Thread-safe manager for the 20x4 I2C LCD."""
    def __init__(self):
        self.lcd = None
        self._cache = [""] * Config.VISIBLE_LINES
        self.lock = threading.RLock()
        
        if USE_LCD:
            try:
                self.lcd = CharLCD('PCF8574', 0x27, cols=Config.LCD_COLS, rows=Config.VISIBLE_LINES)
                self.lcd.backlight_enabled = True
                self.lcd.clear()
            except Exception as e:
                print(f"[Hardware] LCD Init Error: {e}")

    def set_backlight(self, state: bool):
        if self.lcd and self.lcd.backlight_enabled != state:
            self.lcd.backlight_enabled = state

    def clear(self):
        with self.lock:
            if self.lcd:
                self.lcd.clear()
                self._cache = [""] * Config.VISIBLE_LINES

    def write_line(self, row: int, text: str):
        if not self.lcd: return
        txt = (text or "")[:Config.LCD_COLS].ljust(Config.LCD_COLS)
        
        with self.lock:
            if self._cache[row] == txt: return # Caching verhindert unnötigen I2C Traffic
            try:
                self.lcd.cursor_pos = (row, 0)
                self.lcd.write_string(txt)
                self._cache[row] = txt
            except Exception: pass

    def create_char(self, index, bitmap):
        if self.lcd: self.lcd.create_char(index, bitmap)

    def write_custom_chars(self, row, col, chars_list):
        """Schreibt eine Liste von Custom-Char-IDs auf das Display."""
        if not self.lcd: return
        with self.lock:
            try:
                self.lcd.cursor_pos = (row, col)
                for c in chars_list:
                    self.lcd.write_string(" " if c == 32 else chr(c))
                self.lcd.write_string(" ") # Abstand danach
            except Exception: pass

    def close(self):
        with self.lock:
            if not self.lcd:
                return
            try:
                self.lcd.clear()
                self.lcd.backlight_enabled = False
            except Exception:
                pass
            try:
                self.lcd.close(clear=False)
            except TypeError:
                try:
                    self.lcd.close()
                except Exception:
                    pass
            except Exception:
                pass
            self.lcd = None
            self._cache = [""] * Config.VISIBLE_LINES

class RTCManager:
    """Manager for the DS3231 Real Time Clock."""
    def __init__(self):
        self.bus = None
        if SMBUS_AVAILABLE and smbus:
            try: self.bus = smbus.SMBus(1)
            except Exception: pass

    def sync_time(self, n):
        """Synchronisiert ein datetime-Objekt in den RTC Chip."""
        if not self.bus: return
        def d_bcd(v): return ((v // 10) << 4) + (v % 10)
        try:
            data = [d_bcd(n.second), d_bcd(n.minute), d_bcd(n.hour), 
                    n.weekday() + 1, d_bcd(n.day), d_bcd(n.month), d_bcd(n.year % 100)]
            self.bus.write_i2c_block_data(0x68, 0x00, data)
        except Exception: pass

    def close(self):
        if self.bus and hasattr(self.bus, "close"):
            try:
                self.bus.close()
            except Exception:
                pass
        self.bus = None
