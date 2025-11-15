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
AUGMENT_PER_AUDIO = 20  # Weitere Augmentierungen pro Originalaudio

# ===============================
# LOAD BACKGROUND NOISE
# ===============================
preloaded_noise = []
if os.path.exists(BACKGROUND_NOISE_DIR):
    for f in os.listdir(BACKGROUND_NOISE_DIR):
        if f.endswith((".wav", ".mp3")):
            waveform, sr = torchaudio.load(os.path.join(BACKGROUND_NOISE_DIR, f))
            if sr != SAMPLE_RATE:
                waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
            preloaded_noise.append(waveform)
print(f"Loaded {len(preloaded_noise)} background noises.")

# ===============================
# AUDIO AUGMENTER
# ===============================
class AudioAugmenter:
    def __init__(self, preloaded_noise):
        self.preloaded_noise = preloaded_noise
        self.freq_mask = T.FrequencyMasking(freq_mask_param=10)
        self.time_mask = T.TimeMasking(time_mask_param=10)

    def add_white_noise(self, waveform, strength=0.005):
        return waveform + torch.randn_like(waveform) * strength


    def add_background_noise_hq(self, waveform, noise, sr=SAMPLE_RATE, snr_db=25):
        target_len = waveform.size(1)

        # --- 1) Noise auf mono ---
        if noise.size(0) > 1:
            noise = noise.mean(dim=0, keepdim=True)

        # --- 2) Noise auf 1.1 Sekunden schneiden ---
        one_sec = int(sr * 1.1)  # 1.1 Sekunden

        if noise.size(1) >= one_sec:
            noise = noise[:, :one_sec]
        else:
            repeat = (one_sec // noise.size(1)) + 1
            noise = noise.repeat(1, repeat)[:, :one_sec]

        # --- 3) Noise auf Target-Länge bringen ---
        if noise.size(1) < target_len:
            repeat = (target_len // noise.size(1)) + 1
            noise = noise.repeat(1, repeat)[:, :target_len]
        else:
            noise = noise[:, :target_len]

        # --- 4) RMS-Werte berechnen ---
        rms_signal = torch.sqrt(torch.mean(waveform**2) + 1e-9)
        rms_noise  = torch.sqrt(torch.mean(noise**2) + 1e-9)

        # --- 5) SNR-Faktor berechnen ---
        snr_factor = 10 ** (snr_db / 20)

        # Ziel-RMS für das Noise basierend auf gewünschtem SNR
        target_rms_noise = rms_signal / snr_factor

        # --- 6) Noise skalieren ---
        noise = noise * (target_rms_noise / rms_noise)

        # --- 7) Mischung ---
        mixed = waveform + noise

        # --- 8) Clipping verhindern ---
        max_val = mixed.abs().max()
        if max_val > 1.0:
            mixed = mixed / max_val

        return mixed

    def change_gain(self, waveform, min_db=-3, max_db=3, deterministic=False):
        if deterministic:
            db = 0.0
        else:
            db = random.uniform(min_db, max_db)
        gain = 10 ** (db / 20)
        return waveform * gain

    def spec_augment(self, mel_spec, deterministic=False):
        if deterministic:
            return mel_spec
        mel_spec = self.freq_mask(mel_spec)
        mel_spec = self.time_mask(mel_spec)
        return mel_spec

    # Zufällige Augmentierung für Training (außer YT Audio)
    def augment_train(self, waveform):
        if random.random() < 0.8:
            waveform = self.add_white_noise(waveform)
        if random.random() < 0.5:
            waveform = self.change_gain(waveform)
        return waveform

    # Deterministische Augmentierung für Val/Test
    def augment_eval(self, waveform, noise_idx):
        noise = self.preloaded_noise[noise_idx % len(self.preloaded_noise)]
        waveform = self.add_background_noise(waveform, noise)
        waveform = self.change_gain(waveform, deterministic=True)
        waveform = self.add_white_noise(waveform, strength=0.002)
        return waveform

augmenter = AudioAugmenter(preloaded_noise)

# ===============================
# HELPERS
# ===============================
def process_audio(file_path):
    waveform, sr = torchaudio.load(file_path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    target_len = int(TARGET_DURATION * SAMPLE_RATE)
    if waveform.size(1) < target_len:
        waveform = torch.nn.functional.pad(waveform, (0, target_len - waveform.size(1)))
    elif waveform.size(1) > target_len:
        waveform = waveform[:, :target_len]
    return waveform

def to_mel_spec(waveform, deterministic=False):
    mel = T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=400, hop_length=160, n_mels=64)
    mel_spec = mel(waveform)
    mel_spec = torchaudio.functional.amplitude_to_DB(mel_spec, multiplier=10, amin=1e-10, db_multiplier=0)
    mel_spec = augmenter.spec_augment(mel_spec, deterministic)
    return mel_spec

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            folder = os.path.join(OUTPUT_DIR, split, cls)
            os.makedirs(folder, exist_ok=True)

    for cls in CLASSES:
        files = glob(os.path.join(SOURCE_DIR, cls, "*.wav"))

        # YT Audio nur bei not_wake_word
        yt_files = []
        if cls == "not_wake_word":
            yt_files = glob(os.path.join(YT_AUDIO_DIR, "*.wav"))

        random.shuffle(files)
        n = len(files)
        n_train = int(0.90 * n)
        n_val_test = n - n_train
        n_val = n_val_test // 2
        n_test = n_val_test - n_val

        splits = {
            "train": files[:n_train],
            "val": files[n_train:n_train+n_val],
            "test": files[n_train+n_val:]
        }

        for split_name, split_files in splits.items():
            out_folder = os.path.join(OUTPUT_DIR, split_name, cls)
            print(f"Processing {split_name}/{cls} -> {len(split_files)} files")
            for f_idx, f in enumerate(tqdm(split_files)):
                waveform = process_audio(f)
                base = os.path.splitext(os.path.basename(f))[0]

                # Original speichern
                mel_spec = to_mel_spec(waveform, deterministic=(split_name != "train"))
                out_path = os.path.join(out_folder, f"{base}.pt")
                torch.save(mel_spec, out_path)

                # Background Noise nur für SOURCE_DIR Audios, nicht YT Audio
                if split_name == "train" or split_name in ["val", "test"]:
                    for idx, noise in enumerate(preloaded_noise):
                        aug_wave = augmenter.add_background_noise_hq(waveform.clone(), noise)
                        mel_aug = to_mel_spec(aug_wave, deterministic=(split_name != "train"))
                        aug_path = os.path.join(out_folder, f"{base}_bg{idx}.pt")
                        torch.save(mel_aug, aug_path)

            # YT Audio nur Original speichern
            for yt_f in yt_files:
                waveform = process_audio(yt_f)
                base = os.path.splitext(os.path.basename(yt_f))[0]
                mel_spec = to_mel_spec(waveform, deterministic=True)
                out_path = os.path.join(out_folder, f"{base}.pt")
                torch.save(mel_spec, out_path)

