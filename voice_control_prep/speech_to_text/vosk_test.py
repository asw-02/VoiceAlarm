import pyaudio
from vosk import Model, KaldiRecognizer
import json
import sys

# === CONFIGURATION ===

# Path to the VOSK German model
MODEL_PATH = "voice_control/speech_to_text/vosk-model-de-0.21"
SAMPLING_RATE = 16000
AUDIO_CHUNK_SIZE = 4096

# Grammar phrases to improve recognition accuracy
WECHSEL_PHRASEN = [
    "stelle den wecker auf", "stelle den alarm auf",
    "stell den alarm auf", "wecker auf",
    "alarm auf", "mach den wecker auf", "wecker bitte auf",
    "einen alarm für", "aktiviere den alarm für", "bitte den alarm auf",

    "weck mich um", "ich muss um", "ich will um",
    "kannst du mich bitte um", "ich möchte morgen um",
    "ich brauche einen alarm um",

    "wecker ausschalten", "alarm ausschalten", "wecker löschen",
    "alarm löschen", "wecker stoppen", "alarm stoppen",
    "stoppe den alarm", "mach den wecker aus",
    "schalte den alarm ab", "alarm deaktivieren",
    "alle alarme löschen", "wecker abbrechen",

    "stelle den wecker auf halb", "weck mich um halb",
    "aktiviere wecker um halb", "wecker auf viertel vor",
    "wecker auf viertel nach", "stelle wecker auf zehn nach",
    "stelle wecker auf zwanzig vor",
]

# Single words used in commands
EINZEL_WOERTER = [
    "wecker", "alarm", "stelle", "stell", "setzen", "mach", "an", "aus",
    "bitte", "einen", "einstellen", "aktiviere", "uhr", "kannst", "du",
    "den", "schalte", "ab", "ruhe", "jetzt", "von", "am", "vor", "nach",
    "halb", "viertel", "zehn", "zwanzig", "lösche", "nimm", "raus",
    "entferne", "tag", "montag", "dienstag", "mittwoch", "donnerstag",
    "freitag", "samstag", "sonntag",
]

# Numbers from 0 to 59 (spoken)
ZAHLEN = [
    "null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben",
    "acht", "neun", "zehn", "elf", "zwölf", "dreizehn", "vierzehn",
    "fünfzehn", "sechzehn", "siebzehn", "achtzehn", "neunzehn",
    "zwanzig", "einundzwanzig", "zweiundzwanzig", "dreiundzwanzig",
    "vierundzwanzig", "dreißig", "vierzig", "fünfundvierzig",
    "fünfzig", "neunundfünfzig"
]

GRAMMAR = WECHSEL_PHRASEN + EINZEL_WOERTER + ZAHLEN


def initialize_recognizer():
    """Load the Vosk model and create the recognizer."""
    try:
        print(f"Loading model from '{MODEL_PATH}'...")
        model = Model(MODEL_PATH)
        return KaldiRecognizer(model, SAMPLING_RATE, json.dumps(GRAMMAR))
    except Exception as e:
        print(f"Failed to load Vosk model: {e}")
        sys.exit(1)


def listen_and_recognize(recognizer, p):
    """Open the audio stream and process speech in real time."""
    stream = None
    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLING_RATE,
            input=True,
            frames_per_buffer=AUDIO_CHUNK_SIZE
        )
        stream.start_stream()

        print("\n=== Ready. Speak now (Ctrl+C to stop) ===\n")

        while True:
            data = stream.read(AUDIO_CHUNK_SIZE, exception_on_overflow=False)

            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text")
                if text:
                    print(f"\rRecognized: {text}".ljust(80))
            else:
                partial = json.loads(recognizer.PartialResult()).get("partial", "")
                if partial:
                    print(f"\r>> {partial}", end="")

    finally:
        if stream and stream.is_active():
            stream.stop_stream()
            stream.close()


def main():
    recognizer = initialize_recognizer()
    p = pyaudio.PyAudio()

    try:
        listen_and_recognize(recognizer, p)
    except KeyboardInterrupt:
        print("\nStopping program...")
    finally:
        print("Releasing PyAudio resources.")
        p.terminate()


if __name__ == "__main__":
    main()
