import pyaudio
from vosk import Model, KaldiRecognizer
import json
import sys

# === KONFIGURATION ===

# Pfad zu Ihrem VOSK-Modell (Deutsch)
MODEL_PATH = "voice_control/speech_to_text/vosk-model-de-0.21" 
SAMPLING_RATE = 16000
AUDIO_CHUNK_SIZE = 4096 

# --- ERWEITERTE GRAMMATIK FÜR HÖHERE GENAUIGKEIT ---
# Phrasen erhöhen die Wahrscheinlichkeit für die richtigen Befehle
WECHSEL_PHRASEN = [
    # Wichtigste Einstell-Phrasen
    "stelle den wecker auf", "stelle den alarm auf", 
    "stell den alarm auf", "wecker auf", 
    "alarm auf", "mach den wecker auf", "wecker bitte auf",
    "einen alarm für", "aktiviere den alarm für", "bitte den alarm auf",
    
    # Wecken-Phrasen
    "weck mich um", "ich muss um", "ich will um", "kannst du mich bitte um", 
    "ich möchte morgen um", "ich brauche einen alarm um", 
    
    # Lösch-Phrasen
    "wecker ausschalten", "alarm ausschalten", "wecker löschen", 
    "alarm löschen", "wecker stoppen", "alarm stoppen", "stoppe den alarm",
    "mach den wecker aus", "schalte den alarm ab", "alarm deaktivieren", 
    "alle alarme löschen", "wecker abbrechen",
    
    # Relative Zeiten
    "stelle den wecker auf halb", "weck mich um halb", "aktiviere wecker um halb",
    "wecker auf viertel vor", "wecker auf viertel nach",
    "stelle wecker auf zehn nach", "stelle wecker auf zwanzig vor",
]

EINZEL_WOERTER = [
    # Steuerwörter & Nomen
    "wecker", "alarm", "stelle", "stell", "setzen", "mach", "an", "aus", 
    "bitte", "einen", "einstellen", "aktiviere", "uhr", "kannst", "du", 
    "den", "schalte", "ab", "ruhe", "jetzt", "von", "am", "vor", "nach", 
    "halb", "viertel", "zehn", "zwanzig", "lösche", "nimm", "raus", 
    "entferne", "tag", "montag", "dienstag", "mittwoch", "donnerstag", 
    "freitag", "samstag", "sonntag",
]

# Zahlen von 0 bis 59 (als Text)
ZAHLEN = [
    "null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", 
    "acht", "neun", "zehn", "elf", "zwölf", "dreizehn", "vierzehn", 
    "fünfzehn", "sechzehn", "siebzehn", "achtzehn", "neunzehn", 
    "zwanzig", "einundzwanzig", "zweiundzwanzig", "dreiundzwanzig", 
    "vierundzwanzig", "dreißig", "vierzig", "fünfundvierzig", "fünfzig", 
    "neunundfünfzig" 
]

GRAMMAR = WECHSEL_PHRASEN + EINZEL_WOERTER + ZAHLEN


# === HAUPTFUNKTIONEN ===

def initialize_recognizer():
    """Läd das Vosk-Modell und den KaldiRecognizer."""
    try:
        print(f"Lade Modell aus '{MODEL_PATH}'...")
        model = Model(MODEL_PATH)
        rec = KaldiRecognizer(model, SAMPLING_RATE, json.dumps(GRAMMAR))
        return rec
    except Exception as e:
        print(f"Fehler beim Laden des Vosk-Modells: {e}")
        print("Stellen Sie sicher, dass der Pfad korrekt ist und das Modell existiert.")
        sys.exit(1)

def listen_and_recognize(recognizer, p):
    """Öffnet den Audio-Stream und verarbeitet die Daten in Echtzeit."""
    stream = None
    try:
        # Öffnen des Audio-Streams (Standard-Eingabegerät)
        stream = p.open(format=pyaudio.paInt16, 
                        channels=1, 
                        rate=SAMPLING_RATE, 
                        input=True, 
                        frames_per_buffer=AUDIO_CHUNK_SIZE)
        stream.start_stream()

        print("\n=== Bereit. Sprechen Sie jetzt (Strg+C zum Beenden) ===\n")

        while True:
            # Lese Audio-Daten; exception_on_overflow=False verhindert Programmabbruch bei Verzögerung
            data = stream.read(AUDIO_CHUNK_SIZE, exception_on_overflow=False)
            
            if len(data) == 0:
                continue

            if recognizer.AcceptWaveform(data):
                # Fertiger Satz erkannt
                result = json.loads(recognizer.Result())
                text = result.get('text')
                
                if text: 
                    # \r und ljust(80) überschreiben die PartialResult-Anzeige sauber
                    print(f"\rErkannt: {text}".ljust(80)) 
            else:
                # Teilergebnisse für "Echtzeit-Gefühl" anzeigen
                partial_result = json.loads(recognizer.PartialResult())
                partial = partial_result.get('partial', '')
                
                if partial:
                    # \r überschreibt die aktuelle Zeile
                    print(f"\r>> {partial}", end="") 

    finally:
        # Sauberes Schließen des Streams
        if stream and stream.is_active():
            stream.stop_stream()
            stream.close()


def main():
    # 1. Vosk initialisieren
    recognizer = initialize_recognizer()

    # 2. PyAudio initialisieren
    p = pyaudio.PyAudio()
    
    try:
        # 3. Hauptschleife starten
        listen_and_recognize(recognizer, p)
        
    except KeyboardInterrupt:
        print("\n\nBeende Programm...")
    
    finally:
        # 4. Sauberes Aufräumen der PyAudio-Ressourcen
        print("PyAudio-Ressourcen werden freigegeben.")
        p.terminate()

if __name__ == "__main__":
    main()