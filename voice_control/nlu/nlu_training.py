import re
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, classification_report

# Configuration 
TRAINING_FILE = "voice_control/nlu/training_data.json"
MODEL_PATH = "voice_control/nlu/nlu_model.pkl"
CONFIDENCE_THRESHOLD = 0.55  # threshold for "unknown"

# Number mapping 
WORD_TO_NUM = {
    "eins": 1, "ein": 1, "eine": 1, "einer": 1,
    "zwei": 2, "drei": 3, "vier": 4,
    "fünf": 5, "sechs": 6, "sieben": 7, "acht": 8, "neun": 9, "zehn": 10,
    "elf": 11, "zwölf": 12, "dreizehn": 13, "vierzehn": 14, "fünfzehn": 15,
    "sechzehn": 16, "siebzehn": 17, "achtzehn": 18, "neunzehn": 19,
    "zwanzig": 20, "dreißig": 30, "vierzig": 40, "fünfzig": 50
}

STOPWORDS = {"der", "die", "das", "den", "um", "bitte", "mich", "für", "auf", "setzen", "stelle"}


def text_to_int(text):
    """Convert German number word or digit to integer"""
    text = text.lower().strip()
    if text.isdigit():
        return int(text)
    return WORD_TO_NUM.get(text, None)


def preprocess(text):
    """Lowercase, remove punctuation, remove stopwords, convert numbers"""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    tokens = [str(text_to_int(tok)) if text_to_int(tok) is not None else tok for tok in tokens if tok not in STOPWORDS]
    return " ".join(tokens)


# Time patterns 
TIME_PATTERNS = [
    (r"halb\s+(\d{1,2})", lambda m: ((int(m.group(1))-1)%24, 30)),
    (r"viertel vor\s+(\d{1,2})", lambda m: ((int(m.group(1))-1)%24, 45)),
    (r"viertel nach\s+(\d{1,2})", lambda m: (int(m.group(1)), 15)),
    (r"(\d{1,2})[:\.](\d{2})", lambda m: (int(m.group(1)), int(m.group(2)))),
    (r"(\d{1,2}) uhr\s*(\d{1,2})?", lambda m: (int(m.group(1)), int(m.group(2) or 0))),
    (r"(\d{1,2}) vor (\d{1,2})", lambda m: ((int(m.group(2))-1)%24, 60-int(m.group(1)))),
    (r"(\d{1,2}) nach (\d{1,2})", lambda m: (int(m.group(2)), int(m.group(1))))
]

# Weekday mapping (also today/tomorrow semantic meaning)
WEEKDAY_MAP = {
    "montag": 0, "dienstag": 1, "mittwoch": 2, "donnerstag": 3,
    "freitag": 4, "samstag": 5, "sonntag": 6,
    "heute": "PLUS_0", "morgen": "PLUS_1", "übermorgen": "PLUS_2"
}

# Daytime mapping 
DAYTIME_RANGES = {
    "früh": (5, 10),
    "morgens": (6, 10),
    "vormittags": (9, 12),
    "mittags": (12, 13),
    "nachmittags": (13, 16),
    "abends": (16, 22),
    "abend": (16, 22),
    "nachts": (22, 5),
    "nacht": (22, 5)
}

# Helper functions 
def text_to_int(text):
    text = text.lower().strip()
    if text.isdigit():
        return int(text)
    return WORD_TO_NUM.get(text, None)

def preprocess(text):
    """Lowercase, remove punctuation, remove stopwords, replace numbers"""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    tokens = [str(text_to_int(tok)) if text_to_int(tok) is not None else tok for tok in tokens if tok not in STOPWORDS]
    return " ".join(tokens)


class AlarmNLU:
    def __init__(self):
        """
        Initialize the NLU model pipeline using TF-IDF vectorizer and Logistic Regression.
        """
        print("Initializing NLU...")
        self.classifier = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), max_features=3000),
            LogisticRegression(C=10, random_state=42, max_iter=1000, class_weight='balanced')
        )
        self.is_trained = False
        self.model_path = MODEL_PATH

    def load_and_train(self, training_file=TRAINING_FILE):
        """
        Load training data from JSON file, preprocess it, and train the classifier.
        Returns True if training succeeded.
        """
        if not os.path.exists(training_file):
            print(f"Error: {training_file} missing.")
            return False

        with open(training_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        print("Preprocessing training data...")
        texts = [preprocess(item["text"]) for item in data]
        intents = [item["intent"] for item in data]

        print(f"Training model with {len(texts)} samples...")
        self.classifier.fit(texts, intents)
        self.is_trained = True
        return True

    def _extract_time(self, text):
        """
        Detects hour and minute from the input text using German time patterns.
        Returns tuple (hour, minute) or (None, None).
        """
        text_proc = " ".join(str(text_to_int(w) or w) for w in text.lower().split())
        for pattern, func in TIME_PATTERNS:
            m = re.search(pattern, text_proc)
            if m:
                return func(m)
        return None, None

    def _extract_weekday(self, text):
        """
        Detects weekday from text based on WEEKDAY_MAP.
        Returns weekday number (0-6) or special PLUS_x string for relative days.
        """
        text = text.lower()
        for word, val in WEEKDAY_MAP.items():
            if word in text:
                return val
        return None

    def _extract_daytime(self, text):
        """
        Detects German daytime words (e.g., 'morgens', 'abends') from text.
        Returns the matched key or None.
        """
        text = text.lower()
        for word, _ in DAYTIME_RANGES.items():
            if word in text:
                return word
        return None

    def _apply_daytime(self, hour, minute, daytime):
        """
        Adjusts the extracted hour based on the detected daytime range.
        Returns adjusted hour (0-23) or None.
        """
        if hour is None or daytime is None:
            return hour, minute
        
        start, end = DAYTIME_RANGES[daytime]
        
        if start > end:
            if not (hour >= start or hour < end):
                hour += 0
        else:
            if not (start <= hour < end):
                hour += 12

        hour = hour % 24
        return hour, minute

    def parse(self, text):
        """
        Main parsing function:
        - Preprocesses text
        - Predicts intent with classifier
        - Extracts slots: hour, minute, weekday
        - Applies daytime adjustment
        Returns dictionary: {text, intent, confidence, slots}.
        """
        if not self.is_trained:
            return {"error": "Model not trained"}

        processed_input = preprocess(text)
        probas = self.classifier.predict_proba([processed_input])[0]
        best_idx = np.argmax(probas)
        max_proba = probas[best_idx]
        predicted_intent = self.classifier.classes_[best_idx]

        final_intent = predicted_intent
        if max_proba < CONFIDENCE_THRESHOLD or predicted_intent == "no_intent":
            final_intent = "unknown"

        hour, minute = self._extract_time(text)
        weekday = self._extract_weekday(text)
        daytime = self._extract_daytime(text)

        hour, minute = self._apply_daytime(hour, minute, daytime)

        return {
            "text": text,
            "intent": final_intent,
            "confidence": float(max_proba),
            "slots": {"hour": hour, "minute": minute, "weekday": weekday}
        }

    # --- Save trained model ---
    def save_model(self):
        """
        Saves the trained model to disk using joblib.
        """
        if not self.is_trained:
            print("Error: model not trained.")
            return
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        print(f"Saving model to '{self.model_path}'...")
        joblib.dump(self.classifier, self.model_path)
        print("Model saved.")


if __name__ == "__main__":
    nlu = AlarmNLU()
    
    # Load training file
    if nlu.load_and_train():
        print(f"\n--- System ready (threshold: {CONFIDENCE_THRESHOLD}) ---")
        
        test_set = [
            # set_alarm
            {"text": "Stell einen Alarm für Montag abend um halb 7 Uhr", "intent": "set_alarm"},
            {"text": "Weck mich in der Früh viertel nach acht", "intent": "set_alarm"},
            {"text": "Wecker auf 14.30 Uhr", "intent": "set_alarm"},
            {"text": "Kannst du mich bitte morgen um 10 Uhr 35 wecken?", "intent": "set_alarm"},
            {"text": "Stelle den Wecker für Freitag um 18 Uhr", "intent": "set_alarm"},
            {"text": "Wecker auf 6 Uhr 15 stellen", "intent": "set_alarm"},
            {"text": "Weck mich um halb neun am Dienstag", "intent": "set_alarm"},
            {"text": "Alarm auf 20:45 Uhr setzen", "intent": "set_alarm"},
            {"text": "Stell den Alarm um 12 Uhr mittags", "intent": "set_alarm"},
            {"text": "Wecker bitte um 7 Uhr 50 einstellen", "intent": "set_alarm"},
            {"text": "Weck mich um 5 Uhr", "intent": "set_alarm"},
            {"text": "Wecker auf 13 Uhr 30 stellen", "intent": "set_alarm"},
            {"text": "Sorg dafür, dass ich morgen um 6 Uhr wach bin", "intent": "set_alarm"},
            {"text": "Stell sicher, dass ich um halb sieben aufstehe", "intent": "set_alarm"},
            # delete_alarm
            {"text": "Lösche den Wecker am Dienstag", "intent": "delete_alarm"},
            {"text": "Wecker für Montag löschen", "intent": "delete_alarm"},
            {"text": "Alarm auf Freitag entfernen", "intent": "delete_alarm"},
            {"text": "Bitte lösche meinen Wecker für 7 Uhr", "intent": "delete_alarm"},
            {"text": "Lösche den Wecker um 14:00 Uhr", "intent": "delete_alarm"},
            {"text": "Alarm am Mittwoch löschen", "intent": "delete_alarm"},
            {"text": "Nimm den Alarm für morgen früh wieder raus", "intent": "delete_alarm"},
            
            # unknown
            {"text": "Ich trinke gerne Milch", "intent": "unknown"},
            {"text": "Morgen gehe ich um halb 25 Uhr joggen", "intent": "unknown"},
            {"text": "Welches Spiel spielst du?", "intent": "unknown"},
            {"text": "Kannst du mir ein Rezept geben?", "intent": "unknown"},
            {"text": "Heute ist das Wetter schön", "intent": "unknown"},
            {"text": "Wie spät ist es jetzt?", "intent": "unknown"},
            {"text": "Ich gehe heute um 20 Uhr 30 schwimmen", "intent": "unknown"}, 
            {"text": "Ich brauche ein Taxi", "intent": "unknown"},
            {"text": "Mein Wecker hat heute Morgen leider völlig versagt", "intent": "unknown"},
            {"text": "Der Zug nach Berlin fährt um 12:30 Uhr ab", "intent": "unknown"},
            {"text": "Ich habe mir gestern einen neuen digitalen Wecker gekauft", "intent": "unknown"},
            {"text": "Wie spät ist es jetzt gerade in New York", "intent": "unknown"},
            {"text": "Setze einen Timer für die Nudeln auf 10 Minuten", "intent": "unknown"}
        ]

        # Map "no_intent" to "unknown" for evaluation
        y_true = ["unknown" if item["intent"] == "no_intent" else item["intent"] for item in test_set]
        y_pred = []
        
        print("\n--- Single tests ---")
        for t in test_set:
            res = nlu.parse(t["text"])
            y_pred.append(res["intent"])
            
            print(f"Input: '{t['text']}'")
            print(f"Intent: {res['intent']} ({res['confidence']*100:.1f}%)")
            if res['intent'] != 'unknown':
                print(f"Slots:  {res['slots']}")
            print("-" * 20)

        labels = ["set_alarm", "delete_alarm", "unknown"]

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, labels=labels))

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("NLU Confusion Matrix")
        plt.show()

        print("\n---------------------------------")
        print("Tests done.")
        
        nlu.save_model()
