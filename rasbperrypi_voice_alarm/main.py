#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main entry point for the Smart Alarm Clock.
Glues together the database, hardware, UI, and voice threads.
"""

import time
import torch
from config import Config

# --- Components ---
from audio.tts_service import TTSService
from core.database import Database
from hardware.sensors import LightSensor
from ui.manual_setup import ManualStartTime
from ui.clock_ui import AlarmClockUI
from voice_control.voice_thread import VoiceControlThread

# Optimize PyTorch for CPU usage
torch.set_num_threads(2)
torch.set_grad_enabled(False)

if __name__ == "__main__":
    print(">>> System Start...")
    voice_thread = None
    tts = None
    
    # 1. Initialize Database
    db = Database()
    
    # 2. Initialize Hardware Sensors
    sensor = LightSensor(pin=Config.PIN_LIGHT_SENSOR, enabled=False) # Set enable=True if in use
    
    # 3. Manual Time Set (Blocking UI on Startup)
    m_start = ManualStartTime()
    start_data = m_start.run() 

    # 4. Initialize TTS
    tts = TTSService()
    
    # 5. Initialize Main UI
    ui = AlarmClockUI(db, start_data, light_sensor=sensor, tts_service=tts)
    
    # 6. Initialize and Start Voice Thread
    voice_thread = VoiceControlThread(ui, tts_service=tts)
    voice_thread.start()

    # 7. Main Application Loop
    try:
        last_sec = -1
        last_marquee_tick = 0
        
        while not ui._stop_event.is_set():
            ui.alarm_manager.check() 
            ui.check_buttons()
            now = ui.get_now()

            # Protect state reads with lock
            with ui.state_lock: 
                current_frame_copy = ui.current_frame
                has_temp_msg = ui.temp_msg_content is not None
                msg_expired = time.time() > ui.temp_msg_end
            
            # Drawing logic: Update seconds only on the start screen
            if ui.current_frame == "start" and now.second != last_sec:
                ui.draw()
                last_sec = now.second
                
            # Animation for the overwrite menu
            elif ui.current_frame == "overwrite_select":
                if time.time() - last_marquee_tick > 0.4:
                    ui.draw(force=False)
                    last_marquee_tick = time.time()

            # Clear temporary messages if expired
            if has_temp_msg and msg_expired:
                with ui.state_lock:
                    ui.temp_msg_content = None
                ui.draw(force=True)

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n>>> Terminated by user.")
    except Exception as e:
        print(f"\n>>> CRITICAL ERROR in Main Loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(">>> Shutting down...")
        if voice_thread:
            voice_thread.stop()
            voice_thread.join(timeout=2.0)
        if tts:
            tts.stop()
        ui.shutdown_hw()
