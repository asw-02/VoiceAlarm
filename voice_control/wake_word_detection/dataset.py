import os
import random
import torch
import torchaudio
import torchaudio.transforms as T
from glob import glob
from tqdm import tqdm

# ===============================
# CONFIG
# ===============================
SOURCE_DIR = "voice_control/wake_word_detection/dataset/collect_audio/all_audio"
YT_AUDIO_DIR = "voice_control/wake_word_detection/dataset/yt_audio/not_wake_word"
OUTPUT_DIR = "voice_control/wake_word_detection/dataset/final_dataset"
BACKGROUND_NOISE_DIR = "voice_control/wake_word_detection/dataset/background_noise"

CLASSES = ["wake_word", "not_wake_word"]
SAMPLE_RATE = 16000
TARGET_DURATION = 1.1  # Sekunden

# Hier stellst du ein, wie viele Daten du generieren willst.
# Bei 20: Aus 1 Original werden 21 Dateien (1 Original + 20 Variationen).
AUGMENT_PER_AUDIO = 31  

# ===============================
# LOAD BACKGROUND NOISE
# ===============================
preloaded_noise = []
if os.path.exists(BACKGROUND_NOISE_DIR):
    print("Loading background noise...")
    for f in os.listdir(BACKGROUND_NOISE_DIR):
        if f.endswith((".wav", ".mp3")):
            try:
                path = os.path.join(BACKGROUND_NOISE_DIR, f)
                waveform, sr = torchaudio.load(path)
                if sr != SAMPLE_RATE:
                    waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
                if waveform.size(0) > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                preloaded_noise.append(waveform)
            except Exception:
                pass
print(f"Loaded {len(preloaded_noise)} background noises.")

# ===============================
# AUDIO AUGMENTER
# ===============================
class AudioAugmenter:
    def __init__(self, preloaded_noise):
        self.preloaded_noise = preloaded_noise

    def add_white_noise(self, waveform, strength=0.005):
        return waveform + torch.randn_like(waveform) * strength

    def add_background_noise_hq(self, waveform, snr_db_min=5, snr_db_max=25):
        if not self.preloaded_noise:
            return waveform
            
        # 1. Zufälliges Noise auswählen
        noise = random.choice(self.preloaded_noise)
        
        target_len = waveform.size(1)
        noise_len = noise.size(1)

        # 2. Zufälligen Ausschnitt (Chunk) wählen (WICHTIG für Varianz)
        if noise_len > target_len:
            start = random.randint(0, noise_len - target_len)
            noise_chunk = noise[:, start:start + target_len]
        else:
            repeat = (target_len // noise_len) + 1
            noise_chunk = noise.repeat(1, repeat)[:, :target_len]

        # 3. SNR Berechnung
        rms_signal = torch.sqrt(torch.mean(waveform**2) + 1e-9)
        rms_noise  = torch.sqrt(torch.mean(noise_chunk**2) + 1e-9)
        
        snr_db = random.uniform(snr_db_min, snr_db_max)
        snr_factor = 10 ** (snr_db / 20)
        target_rms_noise = rms_signal / snr_factor

        # 4. Mischen
        noise_chunk = noise_chunk * (target_rms_noise / rms_noise)
        mixed = waveform + noise_chunk

        # 5. Clipping verhindern
        max_val = mixed.abs().max()
        if max_val > 1.0:
            mixed = mixed / max_val

        return mixed

    def change_gain(self, waveform, min_db=-3, max_db=3):
        db = random.uniform(min_db, max_db)
        gain = 10 ** (db / 20)
        return waveform * gain

    def augment_train(self, waveform):
        # Pipeline für Source Audio
        if random.random() < 0.85 and self.preloaded_noise:
            waveform = self.add_background_noise_hq(waveform)
        
        if random.random() < 0.6:
            waveform = self.change_gain(waveform)
            
        if random.random() < 0.4:
            waveform = self.add_white_noise(waveform, strength=0.002)
            
        return waveform

augmenter = AudioAugmenter(preloaded_noise)

# ===============================
# HELPERS
# ===============================
def process_audio(file_path):
    try:
        waveform, sr = torchaudio.load(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
        
    target_len = int(TARGET_DURATION * SAMPLE_RATE)
    
    # Padding / Cutting
    if waveform.size(1) < target_len:
        waveform = torch.nn.functional.pad(waveform, (0, target_len - waveform.size(1)))
    elif waveform.size(1) > target_len:
        # Bei YT Dateien (die schon geschnitten sein sollten), nehmen wir einfach die Mitte oder Anfang
        # Um sicher zu gehen, nehmen wir den Anfang (da user sagt sie sind schon gesplittet)
        waveform = waveform[:, :target_len]
        
    return waveform

def to_mel_spec(waveform):
    # Offline SpecAugment DEAKTIVIERT (Online im Training ist besser)
    mel = T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=400, hop_length=160, n_mels=64)
    mel_spec = mel(waveform)
    mel_spec = torchaudio.functional.amplitude_to_DB(mel_spec, multiplier=10, amin=1e-10, db_multiplier=0)
    return mel_spec

def get_splits(file_list):
    """Hilfsfunktion um Listen konsistent zu splitten"""
    random.shuffle(file_list)
    n = len(file_list)
    n_train = int(0.85 * n)
    n_val = int(0.10 * n)
    # Rest ist Test (ca 5%)
    return {
        "train": file_list[:n_train],
        "val": file_list[n_train:n_train+n_val],
        "test": file_list[n_train+n_val:]
    }

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    # Seed für Reproduzierbarkeit beim Splitten (optional)
    random.seed(42) 
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

    for cls in CLASSES:
        print(f"--- Processing Class: {cls} ---")
        
        # 1. Source Audio (eigene Aufnahmen)
        source_files = glob(os.path.join(SOURCE_DIR, cls, "*.wav"))
        source_splits = get_splits(source_files)

        # Verarbeite Source Audio (Mit Augmentierung im Training)
        for split_name, files in source_splits.items():
            print(f"  Source -> {split_name}: {len(files)} files")
            for f in tqdm(files, desc=f"Source {split_name}"):
                waveform = process_audio(f)
                if waveform is None: continue
                
                base = os.path.splitext(os.path.basename(f))[0]
                unique_id = str(random.randint(10000, 99999)) # Kollisionsschutz

                # A) Original speichern (immer)
                mel_spec = to_mel_spec(waveform)
                torch.save(mel_spec, os.path.join(OUTPUT_DIR, split_name, cls, f"src_{base}_{unique_id}.pt"))

                # B) Augmentierung (NUR Source & NUR Train)
                for i in range(AUGMENT_PER_AUDIO):
                    aug_wave = augmenter.augment_train(waveform.clone())
                    mel_aug = to_mel_spec(aug_wave)
                    torch.save(mel_aug, os.path.join(OUTPUT_DIR, split_name, cls, f"src_{base}_{unique_id}_aug{i}.pt"))


        # 2. YT Audio (Nur wenn not_wake_word)
        if cls == "not_wake_word" and os.path.exists(YT_AUDIO_DIR):
            yt_files = glob(os.path.join(YT_AUDIO_DIR, "*.wav"))
            yt_splits = get_splits(yt_files)
            
            # Verarbeite YT Audio (KEINE Augmentierung, nur Verteilung)
            for split_name, files in yt_splits.items():
                print(f"  YT -> {split_name}: {len(files)} files")
                for f in tqdm(files, desc=f"YT {split_name}"):
                    waveform = process_audio(f)
                    if waveform is None: continue

                    base = os.path.splitext(os.path.basename(f))[0]
                    unique_id = str(random.randint(10000, 99999))

                    # Nur Original speichern, keine Augmentierungsschleife
                    mel_spec = to_mel_spec(waveform)
                    torch.save(mel_spec, os.path.join(OUTPUT_DIR, split_name, cls, f"yt_{base}_{unique_id}.pt"))