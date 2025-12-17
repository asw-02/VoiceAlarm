import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx  
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torchaudio.transforms as T 

# Import your model
from models import CRNNWakeWord 

# ============================================================
# 0️⃣ Setup & Reproducibility
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
# 1️⃣ Dataset & Helper Functions (Unchanged)
# ============================================================

def load_dataset_split(output_dir="voice_control/wake_word_detection/dataset/final_dataset"):
    splits = ["train", "val", "test"]
    dataset_split = []
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

        if self.augment:
            self.spec_augment = nn.Sequential(
                T.FrequencyMasking(freq_mask_param=15),
                T.TimeMasking(time_mask_param=35)
            )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        mel_spec = torch.load(self.file_paths[idx], weights_only=True)
        if self.mean is not None and self.std is not None:
            mel_spec = (mel_spec - self.mean) / (self.std + 1e-6)
        if self.augment:
            mel_spec = self.spec_augment(mel_spec)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return mel_spec, label

def compute_global_mel_stats(train_paths, train_labels, cache_path, batch_size=64):
    if os.path.exists(cache_path):
        print(f"📂 Statistik geladen: {cache_path}")
        stats = torch.load(cache_path, weights_only=True)
        return stats["mean"], stats["std"]

    print("📊 Berechne Statistik neu (kann dauern)...")
    temp_ds = MelDataset(train_paths, train_labels, mean=None, std=None, augment=False)
    num_workers = 4 if os.name != 'nt' else 0
    temp_loader = DataLoader(temp_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

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
    labels = np.array(labels)
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    if len(class_counts) < 2: return None
    weights = total_samples / (len(class_counts) * class_counts)
    return torch.tensor(weights, dtype=torch.float).to(device)

# ============================================================
# 2️⃣ Visualization (Unchanged)
# ============================================================

def plot_detailed_confusion_matrix(cm, class_names, filename):
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / (cm_sum.astype(float) + 1e-7) 
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
    plt.subplot(1, 2, 1)
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.title("Model Accuracy")
    plt.legend(); plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Model Loss")
    plt.legend(); plt.grid(True)

    path = os.path.join(save_dir, "training_history.png")
    plt.savefig(path)
    plt.close()
    print(f"📈 Trainingskurve gespeichert: {path}")

# ============================================================
# 3️⃣ Training Loop (Unchanged)
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
            outputs = model(X)
            loss = criterion(outputs, y)
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
    precision = precision_score(all_labels, all_preds, average="binary", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="binary", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    return acc, precision, recall, f1, cm

# ============================================================
# 4️⃣ Main (Updated for ONNX Export)
# ============================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Nutze Gerät: {device}")
    
    save_dir = "voice_control/wake_word_detection"
    os.makedirs(save_dir, exist_ok=True)
    
    # Paths for saving
    onnx_path = os.path.join(save_dir, "wake_word_model.onnx")  # <--- ONNX Path
    stats_cache_path = os.path.join(save_dir, "dataset_stats.pt")

    # 1. Load Data
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = load_dataset_split()
    train_mean, train_std = compute_global_mel_stats(train_paths, train_labels, stats_cache_path)

    train_dataset = MelDataset(train_paths, train_labels, mean=train_mean, std=train_std, augment=True)
    val_dataset   = MelDataset(val_paths, val_labels, mean=train_mean, std=train_std, augment=False)
    test_dataset  = MelDataset(test_paths, test_labels, mean=train_mean, std=train_std, augment=False)

    num_workers = 8 if os.name != 'nt' else 0 
    batch_size = 128

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers>0))
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                              num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers>0))
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                              num_workers=num_workers, pin_memory=True)

    # 2. Initialize Model
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

    # 3. Class Weights & Loss
    class_weights = get_class_weights(train_labels, device)
    print(f"⚖️ Class Weights: {class_weights}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=6, verbose=True)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))

    # 4. Training Loop
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

        scheduler.step(val_acc)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:02d}/{epochs} [LR: {current_lr:.6f}] | "
              f"Train: Loss {train_loss:.4f}, Acc {train_acc:.4f} | "
              f"Val: Loss {val_loss:.4f}, Acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"🛑 Early Stopping ausgelöst! Beste Val Acc: {best_val_acc:.4f}")
                break

    # 5. Final Evaluation
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

    class_names = ["Wake Word (0)", "Not Wake Word (1)"]
    plot_detailed_confusion_matrix(cm, class_names, os.path.join(save_dir, "confusion_matrix_detailed.png"))
    plot_training_history(history["train_loss"], history["train_acc"], 
                          history["val_loss"], history["val_acc"], save_dir)

    # ============================================================
    # 6️⃣ ONNX Export (Optimized for Runtime)
    # ============================================================
    print("\n📦 Exportiere zu ONNX für Raspberry Pi...")
    
    # 1. Move model to CPU (Crucial for Raspberry Pi / ONNX Runtime compatibility)
    model_cpu = model.to("cpu")
    model_cpu.eval()

    # 2. Create Dummy Input on CPU
    # IMPORTANT: Ensure '44' here matches the time frames expected by your model 
    # (based on your training data). If your inference code uses 110, you might 
    # need to update this or allow dynamic axes.
    example_input = torch.randn(1, 1, 64, 110) 

    try:
        torch.onnx.export(
            model_cpu,               # The CPU model
            example_input,           # The CPU dummy input
            onnx_path,               # File path
            export_params=True,      # Save weights inside the file
            opset_version=12,        # Standard version compatible with RPi
            do_constant_folding=True,
            input_names=['input'],   # Naming inputs
            output_names=['output'], # Naming outputs
            dynamic_axes={
                'input': {0: 'batch_size'},  # Batch size can vary
                'output': {0: 'batch_size'}
            }
        )
        print(f"✅ ONNX Modell erfolgreich gespeichert: {onnx_path}")
        print("💡 Tipp: Nutze 'onnxruntime' auf dem Raspberry Pi für 3-5x mehr Performance.")
        
    except Exception as e:
        print(f"⚠️ Fehler beim ONNX Export: {e}")
        # Fallback: Save standard weights
        torch.save(model.state_dict(), os.path.join(save_dir, "model_weights.pth"))

if __name__ == "__main__":
    main()