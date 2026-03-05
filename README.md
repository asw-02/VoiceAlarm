# Annoying Voice Alarm 

A smart alarm clock based on a Raspberry Pi that provides fully local, offline voice control. The system integrates physical hardware components such as an LCD display, buttons, a solenoid, and sensors with a speech-processing pipeline. Voice interaction is implemented using Vosk for speech-to-text processing and ONNX/PyTorch models for wake-word detection and natural language understanding (NLU).

**The Catch:** This is intentionally designed to be a highly annoying alarm clock for heavy sleepers. You cannot turn it off by simply pressing a snooze button or through voice command. To silence the alarm, you must either physically get up and catch a runaway vehicle, or wake up to solve a math puzzle!

## Features

* **Local Voice Control:** Wake-word detection and Intent parsing (NLU) run entirely on-device.
* **Offline Speech-to-Text:** Reliable speech recognition powered by Vosk.
* **Smart UI:** Can be controlled via voice commands or physical buttons alongside a 20x4 I2C LCD display.
* **Ambient Sensor:** Automatic dimming/shut-off of the backlight using a digital light sensor.
* **Failsafe Storage:** Atomic saving (atomic writes) of alarm states prevents corrupted JSON files during sudden power outages.
* **Hardware Feedback:** Status LEDs (Green, Yellow, Red) provide immediate visual feedback on the AI's listening state.
* **The Ultimate Wake-Up Challenge:** The alarm cannot be dismissed normally. You must complete one of two challenges to stop the ringing:
  * **Catch the Vehicle (Solenoid Pre-Alarm):** Exactly 30 seconds before the alarm rings, a solenoid activates and launches a small vehicle from the clock. To stop the alarm, you must get out of bed, find the vehicle, and return it to its base station.
  * **"Master Math":** If you don't use the physical vehicle method, you must solve a randomly generated, complex math puzzle directly on the LCD screen to prove you are fully awake.
---

## Hardware & Wiring

The system runs on a Raspberry Pi 5 (8 GB RAM). The following pin assignments (BCM / GPIO numbers) are defined:

| Component | Type | GPIO Pin / Connection | Note |
| :--- | :--- | :--- | :--- |
| **Menu Button** | Button | GPIO 4 | Pull-up |
| **Up Button** | Button | GPIO 17 | Pull-up |
| **Down Button** | Button | GPIO 27 | Pull-up |
| **Save Button** | Button | GPIO 22 | Pull-up |
| **Auto/Stop Button**| Button | GPIO 5 | Pull-up |
| **Buzzer** | PWM Output | GPIO 12 | Plays the alarm tone |
| **Solenoid** | MOSFET (Output)| GPIO 18 | Triggers the physical pre-alarm |
| **Light Sensor** | Digital Input | GPIO 23 | Pull-up |
| **Status LED: Red** | LED | GPIO 6 | Error / Command not understood |
| **Status LED: Yellow**| LED | GPIO 13 | AI is listening (Recording) |
| **Status LED: Green** | LED | GPIO 19 | Wake-word detected / Success |
| **LCD 20x4** | I2C | SDA / SCL | Address `0x27` (via PCF8574) |
| **RTC Module** | I2C | SDA / SCL | Address `0x68` (e.g., DS3231) |
| **Microphone** | USB | USB-Port | Sample Rate: 44.1 kHz |

---

## Repository Structure

This repository is divided into two main directories to separate the voice controll development from the physical hardware deployment:

### 1. `voice_control/`
This folder contains everything needed to build, train, and test the voice control system from scratch.
* **Wake-Word Detection:** End-to-end training scripts to build the custom wake-word model and export it from PyTorch to ONNX format.
* **Natural Language Understanding (NLU):** Training pipelines for accurate intent classification and slot extraction.
* **Speech-to-Text (STT):** Testing and validation scripts to evaluate the performance of the offline Vosk STT engine.

### 2. `raspberrypi_voice_alarm/`
This directory houses the production-ready Python codebase deployed on the Raspberry Pi. It seamlessly bridges the trained AI models with the physical hardware components.

* **Core System (`main.py`, `config.py`, `database.py`):** Handles system initialization, global hardware configurations, and failsafe, atomic data storage.
* **Hardware & UI (`hardware_manager.py`, `ui_manager.py`):** Manages physical inputs/outputs (LCD menus, buttons, light sensor) and triggers the core alarm mechanisms (buzzer and solenoid).
* **Voice Pipeline (`voice_assistant.py`, `models/`):** The real-time audio engine executing on-device wake-word detection, STT (Vosk), and NLU intent parsing. *(Note: The `models/` directory must be populated with the files generated in the training folder).*

## 🗣️ Supported Voice Commands (Usage)

> **⚠️ Important Note: Language Restriction** > **This project is currently trained and configured exclusively for the German language.** The offline acoustic model (Vosk) and the custom NLU intent parser will only understand and process German speech. 

To interact with the clock, wait for the wake-word detection (green LED) and say your command. The NLU engine is trained to extract intents and dynamic slots from natural German phrases.

**Setting Alarms (Examples):**
* *"Stelle einen Wecker für morgen früh um 7 Uhr."* (Set an alarm for tomorrow morning at 7:00)
* *"Weck mich in 20 Minuten."* (Wake me up in 20 minutes)
* *"Stelle einen Wecker für Wochentags um halb 8."* (Set an alarm for weekdays at 7:30)
* *"Ich brauche einen Wecker für das Wochenende um 9 Uhr."* (I need an alarm for the weekend at 9:00)

**Managing Alarms (Examples):**
* *"Lösche den Wecker um 7 Uhr."* (Delete the 7:00 alarm)
* *"Lösche alle Wecker."* (Delete all alarms)
* *"Lösche den Wecker eins."* (Delete alarm ID 1)