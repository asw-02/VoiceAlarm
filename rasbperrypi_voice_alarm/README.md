# Raspberry Pi Voice Alarm

A modular Raspberry Pi alarm clock with a 20x4 LCD interface, hardware
buttons, GPIO output control, wake-word detection, Vosk speech recognition,
Qwen through Ollama, and Piper text-to-speech.

The project is built for a German voice workflow, but the code and project
documentation are kept in English. Voice commands, confirmations, and spoken
responses are currently German.

## Features

- LCD-based alarm clock UI with hardware button navigation.
- Up to five configurable alarms stored in a local JSON state file.
- GPIO support for buttons, buzzer, LEDs, light sensor, and MOSFET output.
- Wake-word based voice control.
- Dynamic microphone recording that starts on speech and stops after silence.
- Offline German speech recognition with Vosk.
- Local Qwen assistant through Ollama for chat and command parsing.
- Piper-based German speech output.
- Safety layer for mutating voice commands: alarm changes require UI or spoken
  confirmation before they are applied.

## Project Structure

```text
rasbperrypi_voice_alarm/
|-- main.py
|-- config.py
|-- requirements.txt
|-- README.md
|-- core/
|   |-- alarm_manager.py
|   `-- database.py
|-- hardware/
|   `-- sensors.py
|-- ui/
|   |-- clock_ui.py
|   `-- manual_setup.py
|-- tests/
|-- models/
`-- voice_control/
    |-- command_router.py
    |-- qwen_assistant.py
    |-- speech_format.py
    |-- wake_word_detection.py
    |-- voice_thread.py
    |-- nlu.py          # legacy wrapper
    |-- resampler.py    # legacy utility
    `-- tts.py          # legacy wrapper
```

## Main Components

- `main.py` starts the database, hardware abstraction, LCD UI, alarm manager,
  and voice-control thread.
- `config.py` contains GPIO pin mappings, LCD settings, audio settings, model
  paths, Ollama configuration, and Piper configuration.
- `core/` contains the alarm manager and JSON database logic.
- `hardware/` contains GPIO device setup for buttons, LEDs, buzzer, MOSFET, and
  sensors.
- `ui/` contains the clock display, menu flow, manual alarm setup, and voice
  confirmation screens.
- `voice_control/` contains wake-word detection, Vosk transcription, Qwen/Ollama
  integration, speech formatting, Piper output, and safe command routing.
- `tests/` contains unit tests for alarm behavior, command routing, and speech
  formatting.

## Architecture

`VoiceControlThread` waits for the wake word. After activation, it speaks a
short Piper acknowledgement, records the microphone input, stops recording after
detected silence, and transcribes the generated WAV file with Vosk.

The transcribed text is passed to `VoiceCommandRouter`. Simple local commands
such as time, date, next alarm, active alarms, and common alarm actions are
handled directly in Python. Qwen is used only when the local parser needs help
understanding a command or when the user asks a general chat question.

Alarm-changing actions are deliberately guarded. Creating, deleting, activating,
or deactivating alarms does not write directly to `state.json`. Instead, the
router creates a pending UI confirmation. The action is executed only after the
Save button is pressed or after a spoken confirmation such as `ja`. A spoken
negative response such as `nein`, `stop`, or `abbrechen` cancels the pending
action.

The voice dialog remains open briefly after each answer. If no speech is
detected within `LISTEN_TIMEOUT_SECONDS`, voice control ends and the clock
returns to its normal state.

## Hardware

The default GPIO mapping is defined in `Config` inside `config.py`:

```text
PIN_MENU         = 4
PIN_UP           = 27
PIN_DOWN         = 17
PIN_SAVE         = 22
PIN_AUTO         = 5
PIN_BUZZER       = 12
PIN_LIGHT_SENSOR = 23
PIN_MOSFET       = 18
PIN_LED_ROT      = 6
PIN_LED_GELB     = 0
PIN_LED_GRUEN    = 11
```

The UI is configured for a 20x4 I2C LCD:

```text
LCD_COLS      = 20
VISIBLE_LINES = 4
```

If your wiring differs, update the pin values in `config.py` before running the
application on the Raspberry Pi.

## Requirements

- Raspberry Pi with GPIO access.
- Python 3.
- I2C enabled for the LCD.
- Microphone and audio output configured through ALSA.
- Ollama running locally.
- Qwen model pulled into Ollama.
- German Vosk model.
- Piper binary and German Piper voice model.
- Wake-word ONNX model and dataset statistics file.

## Installation

Install system packages for I2C, LCD, and audio support:

```bash
sudo apt-get update
sudo apt-get install python3-smbus i2c-tools portaudio19-dev alsa-utils
```

Create and activate a Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Install and prepare Ollama, then pull the configured Qwen model:

```bash
ollama pull qwen3:1.7b
```

The application expects Ollama at:

```text
http://localhost:11434/api/chat
```

## Configuration

All important runtime paths are configured in `VoiceConfig` inside `config.py`.

Default Vosk path:

```text
/home/oemer/vosk-stt/vosk-model-de-0.21
```

Default Piper paths:

```text
/home/oemer/piper-tts/piper/piper
/home/oemer/piper-tts/de_DE-thorsten-medium.onnx
```

Default wake-word paths:

```text
/home/oemer/wake-word-detection/wake_word_model.onnx
/home/oemer/wake-word-detection/dataset_stats.pt
```

If your models or binaries are stored somewhere else, update these values:

```python
VoiceConfig.VOSK_MODEL_PATH
VoiceConfig.PIPER_BIN
VoiceConfig.PIPER_MODEL
VoiceConfig.WAKE_MODEL_PATH
VoiceConfig.WAKE_STATS_PATH
```

Microphone and recording behavior can also be tuned in `VoiceConfig`:

```python
VoiceConfig.MIC_DEVICE
VoiceConfig.SAMPLE_RATE
VoiceConfig.START_RMS
VoiceConfig.STOP_RMS
VoiceConfig.SILENCE_SECONDS
VoiceConfig.LISTEN_TIMEOUT_SECONDS
```

## Running the Application

Start the full alarm clock application:

```bash
python3 main.py
```

## Autostart on Raspberry Pi

Use a `systemd` service if the alarm clock should start automatically after the
Raspberry Pi boots. `main.py` starts the voice-control thread itself, so Vosk is
loaded automatically as long as `VoiceConfig.VOSK_MODEL_PATH` points to the
installed Vosk model.

First make sure the project works manually:

```bash
cd /home/oemer/rasbperrypi_voice_alarm
source venv/bin/activate
python3 main.py
```

If your project is stored in a different folder, replace
`/home/oemer/rasbperrypi_voice_alarm` in the commands below.

Create a service file:

```bash
sudo nano /etc/systemd/system/voice-alarm.service
```

Add this content:

```ini
[Unit]
Description=Raspberry Pi Voice Alarm
After=network-online.target sound.target ollama.service
Wants=network-online.target

[Service]
Type=simple
User=oemer
WorkingDirectory=/home/oemer/rasbperrypi_voice_alarm
Environment=PYTHONUNBUFFERED=1
ExecStart=/home/oemer/rasbperrypi_voice_alarm/venv/bin/python /home/oemer/rasbperrypi_voice_alarm/main.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Reload `systemd`, enable the service, and start it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable voice-alarm.service
sudo systemctl start voice-alarm.service
```

Check whether the alarm clock is running:

```bash
sudo systemctl status voice-alarm.service
```

Show live logs:

```bash
journalctl -u voice-alarm.service -f
```

Restart or stop the service:

```bash
sudo systemctl restart voice-alarm.service
sudo systemctl stop voice-alarm.service
```

After the next reboot, `main.py` should start automatically. Vosk does not need
an extra service because the application imports Vosk and loads the configured
model path during voice recognition.

Run the standalone voice assistant module for debugging microphone, Vosk, Qwen,
and Piper behavior:

```bash
python3 voice_control/qwen_assistant.py
```

## Voice Commands

The voice interface is currently optimized for German commands. Examples:

```text
Wie viel Uhr ist es?
Welches Datum haben wir?
Stelle einen Wecker um halb sieben.
Stelle einen Wecker morgen um 07:30.
Stelle einen Wecker werktags um 06:45.
Welche Wecker sind aktiv?
Wann klingelt der naechste Wecker?
Deaktiviere Wecker eins.
Loesche alle Wecker.
```

Commands that change alarms require confirmation. After the clock asks for
confirmation, say:

```text
ja
nein
abbrechen
stop
```

## State File

Alarms are stored in a local JSON file:

```text
state.json
```

The path is generated by `Config.get_state_path()` and points to the project
directory. The file is created or updated by the database layer at runtime.

## Useful Checks

Compile the main modules:

```bash
python3 -m py_compile config.py voice_control/qwen_assistant.py voice_control/voice_thread.py voice_control/command_router.py ui/clock_ui.py
```

Run the test suite:

```bash
python3 -m unittest discover -s tests
```

## Notes

- The project name folder currently uses the spelling
  `rasbperrypi_voice_alarm`.
- Some legacy wrappers are still present in `voice_control/` for compatibility.
- Qwen never writes alarm data directly. All alarm mutations go through the
  validated command router and UI confirmation flow.
