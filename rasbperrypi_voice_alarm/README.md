

## Project Structure

The project follows Clean Code principles and is highly modular. Here is an overview of the core components:

* `main.py`: The central entry point that initializes and starts all components.
* `config.py`: Contains all configuration variables, file paths, and system thresholds.
* `core/`: Contains the core logic for the alarm clock, such as the database interface (`database.py`) and alarm management logic (`alarm_manager.py`).
* `hardware/`: Includes all scripts governing hardware parts, such as sensors and buzzers (`sensors.py`).
* `ui/`: Encompasses the user interface modules (`clock_ui.py`, `manual_setup.py`) that manage the LCD menu and handling button inputs.
* `voice_control/`: Contains the specific handling of audio recording, resampling (`resampler.py`), background execution (`voice_thread.py`), and intent parsing (`nlu.py`).
* `models/`: (Must be populated manually) Contains the required AI models (Wake Word, NLU, STT).
* `requirements.txt`: Python dependencies.

```text
raspberrypi_voice_alarm/
├── main.py
├── config.py
├── requirements.txt
├── core/
│   ├── alarm_manager.py
│   └── database.py
├── hardware/
│   └── sensors.py
├── models/
├── ui/
│   ├── clock_ui.py
│   └── manual_setup.py
└── voice_control/
    ├── nlu.py
    ├── resampler.py
    └── voice_thread.py
```

## System Architecture

### Layered Architecture
The software is designed following a **Layered Architecture** pattern to ensure high maintainability, scalability, and a clear separation of concerns (Presentation, Business Logic, Voice Processing, Persistence, and Hardware Abstraction).

- **Presentation Layer (`UI`):** Manages user interaction, menu-driven logic, and LCD display outputs (via `AlarmClockUI`). It delegates functional operations to the logic layer without handling complex business logic itself.
- **Business Logic Layer (`Core`):** The functional heart of the system (`AlarmManager`). It manages the alarms, calculates trigger times, and unifies inputs coming from both physical buttons and voice commands.
- **Voice Processing Layer (`Voice Control`):** An independent, concurrent component (`VoiceControlThread`) that handles Wake Word detection, Speech-to-Text (Vosk), and Natural Language Understanding (NLU). It transforms continuous audio data into structured semantic intents and parameters.
- **Persistence Layer (`Core/Database`):** Encapsulates thread-safe JSON-based storage (`state.json`) using synchronization mechanisms (Locks) to prevent data corruption during parallel read/write operations.
- **Hardware Abstraction Layer (`Hardware`):** Encapsulates interaction with physical components (microphone, GPIOs, buzzer, LEDs, LCD). This decouples the business logic from hardware dependencies, increasing future portability.

### Thread Architecture
To guarantee real-time capabilities and responsiveness, the system utilizes a **concurrent execution model**, primarily driven by a dedicated, non-blocking `VoiceControlThread`.

- **Concurrency:** The `VoiceControlThread` runs completely independently from the main UI thread. This ensures that computationally heavy audio inference doesn't block the LCD menu or alarm evaluations.
- **State-based Processing:** The audio thread continuously transitions between a "WAITING" state (listening for the wake word) and a "LISTENING" state (recording and transcribing full sentences).
- **Synchronization:** Since multiple threads must access shared resources (like the `state.json` file or active alarms), `RLock` mechanisms are enforced to protect critical sections and prevent race conditions.
- **Communication:** Voice commands are processed locally within the voice thread and are only handed over to the main business logic once they have been successfully parsed into structured data (Intents + Slots).

---

## Installation & Setup

**1. System Requirements & System Packages**

Make sure I2C is enabled on your Raspberry Pi (`sudo raspi-config` > Interfacing Options > I2C).
Install system dependencies for audio and I2C:

```bash
sudo apt-get update
sudo apt-get install python3-smbus i2c-tools portaudio19-dev
```

**2.Clone Project & Install Dependencies**

It is highly recommended to use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
**3. Download AI Models**

Place the following files and folders into the models/ directory:

* `vosk-model-de-0.21` (Download here https://alphacephei.com/vosk/models)
* `nlu_model.onnx`
* `wake_word_model.onnx`
* `dataset_stats.pt`

**4. Run the Application**

Execute the main script:

```bash
python3 main.py
```

## Autostart on Boot (optional but recommended)

To make this a true standalone alarm clock, you want the script to start automatically as soon as the Raspberry Pi gets power. We can achieve this by creating a `systemd` service.

**1. Create a new service file:**
Open your terminal and run:
```bash
sudo nano /etc/systemd/system/smart-alarm.service
```

**2. Paste the following configuration:**
*(Make sure to adjust the `WorkingDirectory` and `ExecStart` paths if your username is not `pi` or if you placed the folder somewhere else).*

```ini
[Unit]
Description=Smart AI Voice Alarm Clock
After=network.target sound.target

[Service]
# Change 'pi' to your actual Raspberry Pi username if different
User=pi
Group=audio

# The directory where your main.py lives
WorkingDirectory=/home/pi/raspberrypi_voice_alarm

# Use the Python executable from your virtual environment!
ExecStart=/home/pi/raspberrypi_voice_alarm/venv/bin/python main.py

# Restart automatically if the app crashes
Restart=on-failure
RestartSec=10

# Ensures Python output is sent straight to syslog without buffering
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

Save the file (`CTRL+O`, `Enter`) and exit nano (`CTRL+X`).

**3. Enable and start the service:**
Tell the system to recognize your new service, enable it to run on boot, and start it immediately:

```bash
sudo systemctl daemon-reload
sudo systemctl enable smart-alarm.service
sudo systemctl start smart-alarm.service
```

**4. How to check if it's working:**
If you need to see the live console output (e.g., to read your print statements or check for errors), you can view the logs with:
```bash
journalctl -u smart-alarm.service -f
```
*(Press `CTRL+C` to exit the log view).*