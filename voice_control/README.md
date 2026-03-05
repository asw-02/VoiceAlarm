# Voice Control Module
This module is used for **testing and developing** the voice control components of the `VoiceAlarm` project. It provides a sandbox for training, evaluating, and experimenting with the three main sub-modules: Wake Word Detection, Speech-to-Text (STT), and Natural Language Understanding (NLU).

The entire pipeline is built with efficiency in mind and features modern ML ops practices such as hyperparameter tuning and tracking via MLflow.

## Project Structure
```text
voice_control/
├── nlu/                              # Natural Language Understanding pipeline
│   ├── generate_train_data.py        # Script to generate training data from templates
│   ├── nlu_training.py               # NLU model training & evaluation
│   ├── templates.txt                 # Sentence templates for data generation
│   └── training_data.json            # Generated training dataset
├── speech_to_text/                   # Speech-to-text inference scripts
│   ├── vosk-model-de-0.21/           # German Vosk language model
│   ├── vosk_test.py                  # Vosk STT test script
│   └── whisper_test.py               # Faster Whisper STT test script
├── wake_word_detection/              # Wake word detection model training & evaluation
│   ├── dataset/                      # Audio dataset for training
│   ├── models/                       # Model architecture definitions
│   │   ├── __init__.py
│   │   └── crnn.py                   # CRNN model definition
│   ├── best_model.py                 # Best model evaluation script
│   ├── dataset.py                    # Dataset preparation & augmentation
│   ├── train.py                      # Training with Optuna & MLflow
│   └── model_weights.pth             # Trained model weights
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Sub-modules Documentation

### 1. Wake Word Detection (`wake_word_detection/`)
Wake Word Detection is a key technology in modern voice control. It allows a system to permanently "listen" in the background for a specific trigger without continuously having to run complex and computationally intensive speech recognition models. The wake word in this project is **"Hey Wecker"**.

A typical wake-word system acts like an intelligent bouncer and is characterized by the following features:

- **Continuous Audio Buffer:** The microphone captures audio data and stores it in a very short, continuously overwriting buffer. In this implementation, a rolling buffer of approximately 1.1 seconds in length is used.
- **High Resource Efficiency:** To conserve processor power, a very small, optimized model is used for detection. The system uses a special ONNX model that operates quickly and locally on the CPU to specifically search for the pattern of "Hey Wecker".
- **Privacy by Design:** As long as the activation word ("Hey Wecker") is not spoken, the audio data is immediately discarded. Only when a specific detection threshold (in the code, a confidence of over 90%) is exceeded does the system change its state.

**Data Collection & Processing**
To ensure high model performance and robustness, training data was manually recorded using a smartphone (ensuring clear audio without background noise):
- **Positive & Negative Samples:** Includes the wake word ("Hey Wecker") and false-trigger examples (e.g., "Hey Alexa", "Hallo") to minimize false activations.
- **High Variability:** Covers different speaking styles (monotonous, energetic, angry).
- **Heterogeneous Speakers:** Features female/male voices, varying ages, pitches, and speaking speeds.

**Data Augmentation**
To simulate a real bedroom environment and improve model robustness against background noise, the dataset is heavily augmented:
- **Offline (Audio Level):** Each clean recording is automatically multiplied into 31 variations using:
  - **Background Noise (85%):** Mixing in bedroom noises (snoring, footsteps, breathing) at varying volumes (5-25 dB SNR).
  - **Volume Gain (60%):** Randomly adjusting the overall track volume (±3 dB).
  - **White Noise (40%):** Adding static background hiss.
- **Online (Spectrogram Level):** During training, **SpecAugment** is applied directly to the Mel-spectrograms to force the model to learn from incomplete data. This randomly applies time warping, frequency masking, and time masking.

**Architecture & Best Model Parameters**
The model is a Convolutional Recurrent Neural Network (CRNN) built with PyTorch. After optimization, the final model was trained with the following hyperparameters, which provided the best compromise between model size and accuracy (F1-Score):
- **Learning Rate:** 0.00266
- **Weight Decay:** 0.00012
- **Dropout:** 0.24
- **Conv Channels:** 32
- **Hidden Size (GRU):** 128
- **Num Layers (GRU):** 1
- **Batch Size:** 128

**Features:** 
- Extracts and normalizes Mel-Spectrogram features from audio data.
- Utilizes [Optuna](https://optuna.org/) for automated Hyperparameter Optimization.
- Tracks experiments, logs metrics, and logs models using [MLflow](https://mlflow.org/).
- Leverages PyTorch's Automatic Mixed Precision (`torch.amp.autocast`) to accelerate training on compatible GPUs.
- Monitors training progress via visual graphs (confusion matrices, training curves).

### 2. Speech-to-Text (`speech_to_text/`)
Once the wake word is detected, the audio is recorded and transcribed into text locally.
- **Methods available:** Provides test implementations for robust, offline transcription.
  - **Vosk:** Uses the German Vosk model (`vosk-model-de-0.21`) for lightweight and fast real-time decoding.
  - **Faster Whisper:** Uses the `faster-whisper` library for highly accurate transcriptions.

### 3. Natural Language Understanding (`nlu/`)
The transcribed text is then passed to the NLU component to extract user intents (e.g., setting or deleting an alarm) and identify slots (e.g., time, day).

- **Architecture:** TF-IDF Vectorizer combined with a Logistic Regression classifier (`scikit-learn`).
- **Abilities:** 
  - Classifies text into defined intents such as `set_alarm`, `delete_alarm`, and `unknown` as a fallback.
  - Uses Regex-based slot filling and mapping features specifically tailored to the German language (e.g., "halb 7", "viertel vor 8", "übermorgen").
- **Deployment:** The trained model is exported to the **ONNX** format (`skl2onnx`) for fast, dependency-free inference in production.

**Applied Regex Patterns**
The NLU module employs the following regular expressions (Regex) to extract time formats from transcribed speech:

| Context | Regex Pattern | Example Match |
| :--- | :--- | :--- |
| **Punctuation** | `[^\w\s]` | "Hallo, Welt!" → "Hallo Welt" |
| **"Halb" (Half past)** | `halb\s+(\d{1,2})` | "halb 7" → 06:30 |
| **"Viertel vor" (Quarter to)** | `viertel vor\s+(\d{1,2})` | "viertel vor 8" → 07:45 |
| **"Viertel nach" (Quarter past)** | `viertel nach\s+(\d{1,2})` | "viertel nach 8" → 08:15 |
| **Digital Time** | `(\d{1,2})[:\.](\d{2})` | "14:30" or "07.45" |
| **"Uhr" Formats** | `(\d{1,2}) uhr\s*(\d{1,2})?` | "14 uhr 30" → 14:30 |
| **"Vor" (Minutes to)** | `(\d{1,2}) vor (\d{1,2})` | "10 vor 8" → 07:50 |
| **"Nach" (Minutes past)** | `(\d{1,2}) nach (\d{1,2})` | "10 nach 8" → 08:10 |

## Getting Started / How to Use

As a testing and development module, you can run the individual components to train models or test inference. Make sure you have installed the dependencies from the `Requirements` section.

### 1. Training the NLU Model
The NLU model requires a generated dataset built off standard text templates:

```bash
# 1. Generate 30,000 synthetic training samples from templates.txt
python voice_control/nlu/generate_train_data.py

# 2. Train the classifier and export to an optimized ONNX model
python voice_control/nlu/nlu_training.py
```

### 2. Testing Speech-to-Text
You can test the offline audio transcription capabilities using Vosk or Faster Whisper. Make sure your microphone is connected.

```bash
# Test with Vosk (requires the downloaded vosk-model-de-0.21)
python voice_control/speech_to_text/vosk_test.py

# Test with Faster Whisper
python voice_control/speech_to_text/whisper_test.py
```

### 3. Training the Wake Word Model
If you have collected and preprocessed your wake word audio into `wake_word_detection/dataset/`, you can train the model with automated hyperparameter tuning via Optuna and MLflow:

```bash
python voice_control/wake_word_detection/train.py

# To monitor models and hyperparameter progress, launch MLflow:
mlflow ui
```

## Requirements
To run this project, make sure to install all necessary requirements:

```bash
pip install -r requirements.txt
```
Key libraries include PyTorch, Torchaudio, Scikit-learn, ONNX utilities, Vosk, Faster-Whisper, Optuna, MLflow, and Matplotlib.
