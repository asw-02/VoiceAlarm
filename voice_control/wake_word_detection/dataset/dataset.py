import os
import random
import shutil
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from glob import glob
import soundfile as sf

# ===============================================
# CONFIG
# ===============================================
SOURCE_DIR = "voice_control/wake_word_detection/dataset/collect_audio"
OUTPUT_DIR = "voice_control/wake_word_detection/dataset"
CLASSES = ["wake-word", "not-wake-word"]

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

SAMPLE_RATE = 16000
AUGMENT_DOUBLE_DATA = True

# Augmentation settings
NOISE_LEVEL = 0.005
GAIN_DB_RANGE = (-3, 3)  # volume variation in dB
TIME_STRETCH_RANGE = (0.9, 1.1)  # Stretch factor

# ===============================================
# Hilfsfunktion zum Laden von WAV-Dateien
# ===============================================
def load_file_paths(folder):
    paths = []
    labels = []
    for label_idx, class_name in enumerate(CLASSES):
        class_path = os.path.join(folder, class_name)
        if not os.path.exists(class_path):
            continue
        for f in os.listdir(class_path):
            if f.lower().endswith(".wav"):
                paths.append(os.path.join(class_path, f))
                labels.append(label_idx)
    return paths, labels

# ===============================================
# Dataset Klasse NUR FÜR CRNN
# ===============================================
class WakeWordDataset(Dataset):
    def __init__(self, file_paths, labels, sample_rate=SAMPLE_RATE, augment=True):
        self.file_paths = file_paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.augment = augment

        # Mel-Spektrogramm + SpecAugment
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=64
        )
        self.freq_mask = T.FrequencyMasking(freq_mask_param=10)
        self.time_mask = T.TimeMasking(time_mask_param=10)

    def __len__(self):
        return len(self.file_paths)

    def load_audio(self, path):
        waveform, sr = sf.read(path, dtype='float32')
        if len(waveform.shape) == 1:
            waveform = torch.from_numpy(waveform).unsqueeze(0)
        else:
            waveform = torch.from_numpy(waveform).transpose(0, 1)

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        # 1 Sekunde Länge sicherstellen
        target_length = int(self.sample_rate * 1.0)
        if waveform.size(1) < target_length:
            pad_size = target_length - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))
        elif waveform.size(1) > target_length:
            waveform = waveform[:, :target_length]

        return waveform

    def add_noise(self, waveform):
        return waveform + torch.randn_like(waveform) * NOISE_LEVEL

    def change_gain(self, waveform):
        db = random.uniform(*GAIN_DB_RANGE)
        gain = 10 ** (db / 20)
        return waveform * gain

    def time_stretch(self, waveform):
        rate = random.uniform(*TIME_STRETCH_RANGE)
        stretched = torchaudio.functional.time_stretch(
            torchaudio.transforms.Spectrogram()(waveform),
            hop_length=160,
            n_freq=201,
            overriding_rate=rate
        )
        # zurück zu Waveform
        waveform = torchaudio.functional.istft(
            stretched,
            n_fft=400,
            hop_length=160,
            length=int(SAMPLE_RATE * 1.0)
        ).unsqueeze(0)
        return waveform

    def spec_augment(self, spec):
        spec = self.freq_mask(spec)
        spec = self.time_mask(spec)
        return spec

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        waveform = self.load_audio(path)

        if self.augment:
            if random.random() < 0.7:
                waveform = self.add_noise(waveform)
            if random.random() < 0.5:
                waveform = self.change_gain(waveform)
            if random.random() < 0.5:
                waveform = self.time_stretch(waveform)

        mel_spec = self.mel_transform(waveform)
        mel_spec = torchaudio.functional.amplitude_to_DB(
            mel_spec, multiplier=10, amin=1e-10, db_multiplier=0
        )

        if self.augment:
            if random.random() < 0.5:
                mel_spec = self.spec_augment(mel_spec)

        return mel_spec, torch.tensor(label, dtype=torch.long)

# ===============================================
# Funktion: Daten splitten + augmentieren
# ===============================================
def prepare_dataset():
    for c in CLASSES:
        files = glob(os.path.join(SOURCE_DIR, c, "*.wav"))
        random.shuffle(files)
        n = len(files)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        n_test = n - n_train - n_val

        splits = {
            "train": files[:n_train],
            "val": files[n_train:n_train+n_val],
            "test": files[n_train+n_val:]
        }

        for split_name, split_files in splits.items():
            split_path = os.path.join(OUTPUT_DIR, split_name, c)
            os.makedirs(split_path, exist_ok=True)

            # Alte Dateien löschen
            for f in os.listdir(split_path):
                os.remove(os.path.join(split_path, f))

            for f in split_files:
                shutil.copy(f, split_path)

                # Augmentierte Version nur für Training speichern
                if AUGMENT_DOUBLE_DATA and split_name == "train":
                    waveform, sr = torchaudio.load(f)
                    if sr != SAMPLE_RATE:
                        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

                    # Noise + Gain + Stretch
                    waveform_aug = waveform.clone()
                    waveform_aug = waveform_aug + torch.randn_like(waveform_aug) * NOISE_LEVEL

                    if random.random() < 0.5:
                        db = random.uniform(*GAIN_DB_RANGE)
                        gain = 10 ** (db / 20)
                        waveform_aug = waveform_aug * gain

                    if random.random() < 0.5:
                        stretch_factor = random.uniform(*TIME_STRETCH_RANGE)
                        spec = torchaudio.transforms.Spectrogram()(waveform_aug)
                        stretched = torchaudio.functional.time_stretch(
                            spec, hop_length=160, n_freq=201, overriding_rate=stretch_factor
                        )
                        waveform_aug = torchaudio.functional.istft(
                            stretched, n_fft=400, hop_length=160,
                            length=int(SAMPLE_RATE * 1.0)
                        ).unsqueeze(0)

                    base, ext = os.path.splitext(os.path.basename(f))
                    aug_path = os.path.join(split_path, f"{base}_aug{ext}")
                    torchaudio.save(aug_path, waveform_aug, SAMPLE_RATE)

        print(f"{c}: total={n}, train={n_train}, val={n_val}, test={n_test}")

# ===============================================
# Funktion: DataLoader erstellen
# ===============================================
def get_dataloaders(batch_size=32, num_workers=2):
    train_files, train_labels = load_file_paths(os.path.join(OUTPUT_DIR, "train"))
    val_files, val_labels = load_file_paths(os.path.join(OUTPUT_DIR, "val"))
    test_files, test_labels = load_file_paths(os.path.join(OUTPUT_DIR, "test"))

    train_dataset = WakeWordDataset(train_files, train_labels, augment=True)
    val_dataset = WakeWordDataset(val_files, val_labels, augment=False)
    test_dataset = WakeWordDataset(test_files, test_labels, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

# ===============================================
# Main
# ===============================================
if __name__ == "__main__":
    print("\n📂 Preparing dataset for CRNN ...")
    prepare_dataset()

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32, num_workers=2)

    print("\n🧪 Checking one batch for CRNN ...")
    for mel_spec, labels in train_loader:
        print("Mel Spec:", mel_spec.shape)
        print("Labels:", labels[:10])
        break
