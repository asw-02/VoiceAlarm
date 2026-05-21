import random
import json
import os

# --- Configuration ---
TEMPLATE_FILE = "voice_control/nlu/templates.txt"
OUTPUT_FILE = "voice_control/nlu/training_data.json"
NUM_SAMPLES = 30000  # total number of samples

class DataGenerator:
    def __init__(self):
        # Realistic hours for alarm usage
        self.hours = [str(i) for i in range(5, 24)]
        # Commonly used minutes
        self.minutes = ["00", "05", "10", "15", "20", "25", "30", "35", "40", "45", "50", "55"]
        self.days = ["Montag","Dienstag","Mittwoch","Donnerstag","Freitag",
                     "Samstag","Sonntag","morgen","übermorgen"]

        # Class distribution
        self.class_distribution = {"set_alarm": 0.3, "delete_alarm": 0.2, "no_intent": 0.5}

    def load_templates(self, filepath):
        """Load templates from file."""
        templates = {"set_alarm": [], "delete_alarm": [], "no_intent": []}
        current_section = None
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File '{filepath}' not found!")

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("[") and line.endswith("]"):
                    sec = line[1:-1]
                    current_section = sec if sec in templates else None
                    continue
                if current_section:
                    templates[current_section].append(line)
        return templates

    def fill_template(self, tmpl):
        """Replace placeholders with random values."""
        s = tmpl
        if "{stunde}" in s:
            s = s.replace("{stunde}", random.choice(self.hours))
        if "{minute}" in s:
            s = s.replace("{minute}", random.choice(self.minutes))
        if "{wochentag}" in s:
            s = s.replace("{wochentag}", random.choice(self.days))
        return s

    def generate_and_save(self):
        print(f"Loading templates from {TEMPLATE_FILE}...")
        templates = self.load_templates(TEMPLATE_FILE)
        data = []

        print(f"Generating {NUM_SAMPLES} samples...")
        for intent, frac in self.class_distribution.items():
            n_samples = int(NUM_SAMPLES * frac)
            tmpl_list = templates[intent]
            if not tmpl_list:
                continue
            for _ in range(n_samples):
                tmpl = random.choice(tmpl_list)
                text = self.fill_template(tmpl)
                data.append({"text": text, "intent": intent})

        # Shuffle data
        random.shuffle(data)

        # Save file
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Done! Saved {len(data)} samples to {OUTPUT_FILE}.")


if __name__ == "__main__":
    generator = DataGenerator()
    generator.generate_and_save()