import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torchaudio.transforms as T 

# Importiere dein Modell (muss im selben Ordner oder PYTHONPATH liegen)
from models import CRNNWakeWord 

# ============================================================
# 0️⃣ Setup & Reproduzierbarkeit
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ============================================================
# 1️⃣ Dataset & Hilfsfunktionen
# ============================================================

def load_dataset_split(output_dir="voice_control/wake_word_detection/dataset/final_dataset"):
    """Lädt Pfade. ACHTUNG: Die Reihenfolge in der Liste bestimmt das Label ID (0, 1)."""
    splits = ["train", "val", "test"]
    dataset_split = []
    
    # Deine Ordnerstruktur. Index 0 = wake_word, Index 1 = not_wake_word
    class_names = ["wake_word", "not_wake_word"]
    
    for split in splits:
        paths, labels = [], []
        for label_idx, class_name in enumerate(class_names):
            folder = os.path.join(output_dir, split, class_name)
            if not os.path.exists(folder):
                continue
            for f in os.listdir(folder):
                if f.endswith(".pt"):
                    paths.append(os.path.join(folder, f))
                    labels.append(label_idx)
        dataset_split.append((paths, labels))
    return tuple(dataset_split)


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, labels, mean=None, std=None, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.mean = mean
        self.std = std
        self.augment = augment

        # Data Augmentation: Maskiert Frequenz- und Zeitbänder
        if self.augment:
            self.spec_augment = nn.Sequential(
                T.FrequencyMasking(freq_mask_param=15),
                T.TimeMasking(time_mask_param=35)
            )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Laden (weights_only=True für Sicherheit)
        mel_spec = torch.load(self.file_paths[idx], weights_only=True)

        # 1. Normalisierung
        if self.mean is not None and self.std is not None:
            mel_spec = (mel_spec - self.mean) / (self.std + 1e-6)

        # 2. Augmentation (nur im Training aktiv)
        if self.augment:
            mel_spec = self.spec_augment(mel_spec)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return mel_spec, label


def compute_global_mel_stats(train_paths, train_labels, cache_path, batch_size=64):
    """Berechnet Mean/Std über das Training-Set oder lädt Cache."""
    if os.path.exists(cache_path):
        print(f"📂 Statistik geladen: {cache_path}")
        stats = torch.load(cache_path, weights_only=True)
        return stats["mean"], stats["std"]

    print("📊 Berechne Statistik neu (kann dauern)...")
    temp_ds = MelDataset(train_paths, train_labels, mean=None, std=None, augment=False)
    # Num_workers=0 für Statistik oft sicherer gegen Race-Conditions, sonst 4
    temp_loader = DataLoader(temp_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    sum_ = 0.0
    sum_sq_ = 0.0
    num_elements = 0

    for mel_batch, _ in tqdm(temp_loader, desc="Statistik"):
        sum_ += torch.sum(mel_batch)
        sum_sq_ += torch.sum(torch.pow(mel_batch, 2))
        num_elements += mel_batch.nelement()

    mean = sum_ / num_elements
    variance = (sum_sq_ / num_elements) - torch.pow(mean, 2)
    std = torch.sqrt(variance)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save({"mean": mean, "std": std}, cache_path)
    return mean, std


def get_class_weights(labels, device):
    """Gewichtet Klassen entgegengesetzt ihrer Häufigkeit."""
    labels = np.array(labels)
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    if len(class_counts) < 2: 
        return None # Fallback falls nur eine Klasse da ist
    weights = total_samples / (len(class_counts) * class_counts)
    return torch.tensor(weights, dtype=torch.float).to(device)

# ============================================================
# 2️⃣ Visualisierung (Neu: Detailed Confusion Matrix)
# ============================================================

def plot_detailed_confusion_matrix(cm, class_names, filename):
    """Speichert CM mit absoluten Zahlen UND Prozentwerten."""
    # Berechne Prozentzahlen pro Zeile (True Labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / (cm_sum.astype(float) + 1e-7) 

    # Beschriftungen erstellen
    annot = np.empty_like(cm).astype(object)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            annot[i, j] = f"{c}\n({p:.1%})"

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', cbar=True,
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    
    ax.set_title("Confusion Matrix (Count & Percent)")
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"🖼️ Detaillierte Confusion Matrix gespeichert: {filename}")


def plot_training_history(train_losses, train_accs, val_losses, val_accs, save_dir):
    plt.figure(figsize=(12, 5))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.title("Model Accuracy")
    plt.xlabel("Epochs"); plt.ylabel("Accuracy")
    plt.legend(); plt.grid(True)

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Model Loss")
    plt.xlabel("Epochs"); plt.ylabel("Loss")
    plt.legend(); plt.grid(True)

    path = os.path.join(save_dir, "training_history.png")
    plt.savefig(path)
    plt.close()
    print(f"📈 Trainingskurve gespeichert: {path}")

# ============================================================
# 3️⃣ Training Loop & Validation
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        # Mixed Precision Forward
        with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
            outputs = model(X)
            loss = criterion(outputs, y)

        # Scaled Backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * X.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        
    return running_loss / len(loader.dataset), correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            running_loss += loss.item() * X.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return running_loss / len(loader.dataset), correct / total


def evaluate_full(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            preds = model(X).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    acc = (all_preds == all_labels).mean()
    # zero_division=0 verhindert Warnungen, falls eine Klasse gar nicht vorhergesagt wird
    precision = precision_score(all_labels, all_preds, average="binary", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="binary", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    return acc, precision, recall, f1, cm

# ============================================================
# 4️⃣ Main
# ============================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Nutze Gerät: {device}")
    
    # Pfade
    save_dir = "voice_control/wake_word_detection"
    os.makedirs(save_dir, exist_ok=True)
    torchscript_path = os.path.join(save_dir, "wake_word_model.pt")
    stats_cache_path = os.path.join(save_dir, "dataset_stats.pt")

    # 1. Daten laden
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = load_dataset_split()
    
    # Statistik berechnen
    train_mean, train_std = compute_global_mel_stats(train_paths, train_labels, stats_cache_path)

    # Datasets (Augmentierung NUR für Training!)
    train_dataset = MelDataset(train_paths, train_labels, mean=train_mean, std=train_std, augment=True)
    val_dataset   = MelDataset(val_paths, val_labels, mean=train_mean, std=train_std, augment=False)
    test_dataset  = MelDataset(test_paths, test_labels, mean=train_mean, std=train_std, augment=False)

    # DataLoaders (Optimierte Worker Anzahl & Pin Memory)
    # Tipp: Wenn du Windows nutzt und Fehler kriegst, setze num_workers=0
    num_workers = 8 if os.name != 'nt' else 0 
    batch_size = 128

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers>0))
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                              num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers>0))
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                              num_workers=num_workers, pin_memory=True)

    # 2. Modell initialisieren (Beste Hyperparameter)
    best_params = {
        "lr": 0.002662456091517711,
        "dropout": 0.24032061324409332,
        "hidden_size": 128,
        "num_layers": 1,
        "conv_channels": 32,
        "weight_decay": 0.00012278813822580662
    }

    model = CRNNWakeWord(
        num_classes=2,
        conv_channels=best_params["conv_channels"],
        hidden_size=best_params["hidden_size"],
        num_layers=best_params["num_layers"],
        dropout=best_params["dropout"]
    ).to(device)

    # 3. Class Weights berechnen & Loss definieren
    class_weights = get_class_weights(train_labels, device)
    print(f"⚖️ Class Weights: {class_weights}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
    
    # Scheduler: Reduziert LR, wenn Validation nicht besser wird
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=6, verbose=True)
    
    # Scaler für Mixed Precision
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))

    # 4. Training Loop mit Early Stopping
    epochs = 50
    patience = 8
    counter = 0
    best_val_acc = 0.0
    best_model_state = None
    
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print("\n🚀 Starte Training...")
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Learning Rate anpassen basierend auf Val Accuracy
        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:02d}/{epochs} [LR: {current_lr:.6f}] | "
              f"Train: Loss {train_loss:.4f}, Acc {train_acc:.4f} | "
              f"Val: Loss {val_loss:.4f}, Acc {val_acc:.4f}")

        # Early Stopping Check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"🛑 Early Stopping ausgelöst! Beste Val Acc: {best_val_acc:.4f}")
                break

    # 5. Finale Evaluation
    if best_model_state:
        model.load_state_dict(best_model_state)
        print("\n🏆 Bestes Modell geladen.")

    acc, precision, recall, f1, cm = evaluate_full(model, test_loader, device)
    
    print("\n📊 TEST ERGEBNISSE:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Confusion Matrix (Raw):\n{cm}")

    # Plot Confusion Matrix
    # Label 0 = wake_word, Label 1 = not_wake_word (gemäß load_dataset_split)
    class_names = ["Wake Word (0)", "Not Wake Word (1)"]
    plot_detailed_confusion_matrix(cm, class_names, os.path.join(save_dir, "confusion_matrix_detailed.png"))

    # Plot History
    plot_training_history(history["train_loss"], history["train_acc"], 
                          history["val_loss"], history["val_acc"], save_dir)

    # 6. TorchScript Export
    model.eval()
    try:
        example_input = torch.randn(1, 1, 64, 44).to(device)
        traced_model = torch.jit.trace(model, example_input)
        traced_model.save(torchscript_path)
        print(f"💾 Modell gespeichert: {torchscript_path}")
    except Exception as e:
        print(f"⚠️ Fehler beim Speichern von TorchScript: {e}")
        # Fallback: Normales state_dict speichern
        torch.save(model.state_dict(), os.path.join(save_dir, "model_weights.pth"))

if __name__ == "__main__":
    main()