import os
from functools import partial
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import mlflow
from tqdm import tqdm

from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Sicherstellen, dass models.py existiert und importierbar ist
from models import CRNNWakeWord 

# ============================================================
# Dataset-Klasse
# ============================================================
class MelDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, labels, mean=None, std=None):
        self.file_paths = file_paths
        self.labels = labels
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # weights_only=True für Sicherheit beim Laden
        mel = torch.load(self.file_paths[idx], weights_only=True)

        # Globale Normalisierung (falls Stats vorhanden)
        if self.mean is not None and self.std is not None:
            mel = (mel - self.mean) / (self.std + 1e-6)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return mel, label

# ===============================
# COMPUTE GLOBAL MEAN/STD
# ===============================
def compute_global_mel_stats(train_paths, train_labels, cache_path, batch_size=64, num_workers=4):
    """
    Berechnet oder lädt den globalen Mittelwert und die Standardabweichung
    nur anhand der Trainingsdaten.
    """
    if os.path.exists(cache_path):
        print(f"Lade Statistik von {cache_path}")
        stats = torch.load(cache_path, weights_only=True)
        return stats["mean"], stats["std"]

    print("Berechne Statistik (Mittelwert/Std) für den Trainingsdatensatz...")
    
    temp_ds = MelDataset(train_paths, train_labels, mean=None, std=None)
    temp_loader = DataLoader(temp_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    sum_ = 0.0
    sum_sq_ = 0.0
    num_elements = 0

    for mel_batch, _ in tqdm(temp_loader, desc="Statistikberechnung"):
        sum_ += torch.sum(mel_batch)
        sum_sq_ += torch.sum(torch.pow(mel_batch, 2))
        num_elements += mel_batch.nelement()

    mean = sum_ / num_elements
    variance = (sum_sq_ / num_elements) - torch.pow(mean, 2)
    std = torch.sqrt(variance)

    torch.save({"mean": mean, "std": std}, cache_path)
    print(f"Statistik gespeichert. Mean={mean:.4f}, Std={std:.4f}")

    return mean, std

# ============================================================
# Dataset Loader
# ============================================================
def load_dataset_split(output_dir="voice_control/wake_word_detection/dataset/final_dataset"):
    splits = ["train", "val", "test"]
    dataset_split = []

    for split in splits:
        paths, labels = [], []
        for label_idx, class_name in enumerate(["wake_word", "not_wake_word"]):
            folder = os.path.join(output_dir, split, class_name)
            if not os.path.exists(folder):
                continue
            for f in os.listdir(folder):
                if f.endswith(".pt"):
                    paths.append(os.path.join(folder, f))
                    labels.append(label_idx)
        
        # Shuffle, damit nicht alle gleichen Klassen hintereinander kommen
        combined = list(zip(paths, labels))
        random.shuffle(combined)
        if combined:
            paths, labels = zip(*combined)
            dataset_split.append((list(paths), list(labels)))
        else:
            dataset_split.append(([], []))

    return tuple(dataset_split)

# ============================================================
# Training & Eval Helper
# ============================================================
def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        # Mixed Precision Context
        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            outputs = model(X)
            loss = criterion(outputs, y)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * X.size(0)

    return total_loss / len(dataloader.dataset)

def validate_loss_acc(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)

            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            total_loss += loss.item() * X.size(0)

    return total_loss / total, correct / total

def evaluate_f1(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            preds = model(X).argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # zero_division=0 verhindert Warnungen, falls eine Klasse gar nicht vorhergesagt wird
    precision = precision_score(all_labels, all_preds, average="binary", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="binary", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    return f1, precision, recall, cm

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ============================================================
# Optuna Objective
# ============================================================
def objective(trial, train_loader, val_loader, device, epochs=15):
    
    # Nested Run starten, damit jeder Trial sauber in MLflow gruppiert ist
    with mlflow.start_run(nested=True):
        
        # 1. Hyperparameter vorschlagen
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.2, 0.4)
        hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
        num_layers = trial.suggest_int("num_layers", 1, 2)
        conv_channels = trial.suggest_categorical("conv_channels", [32, 64])
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

        # Parameter in MLflow loggen
        mlflow.log_params(trial.params)

        # 2. Modell initialisieren
        model = CRNNWakeWord(
            num_classes=2,
            conv_channels=conv_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        
        # Scaler nur bei CUDA
        scaler = torch.amp.GradScaler("cuda") if "cuda" in str(device) else None

        best_val_f1 = 0.0

        num_params = count_parameters(model)
        print(f"[Trial {trial.number}] Parameteranzahl: {num_params}")
        mlflow.log_param("num_params", num_params)
        trial.set_user_attr("num_params", num_params)
        
        # 3. Training Loop
        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
            val_loss, val_acc = validate_loss_acc(model, val_loader, criterion, device)
            
            # Scheduler basiert auf Validation Loss
            scheduler.step(val_loss)

            # Metriken berechnen
            val_f1, _, _, _ = evaluate_f1(model, val_loader, device)

            print(f"[Trial {trial.number}] Epoch {epoch+1}/{epochs} | "
                  f"TLoss={train_loss:.4f} | VLoss={val_loss:.4f} | VF1={val_f1:.4f}")

            # Optuna Pruning (Abbruch, wenn Trial schlecht läuft)
            trial.report(val_f1, epoch)
            if trial.should_prune():
                mlflow.set_tag("status", "pruned")
                raise optuna.TrialPruned()

            # WICHTIG: Den besten F1-Wert des gesamten Laufs merken
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
        
        # Am Ende des Trials den besten Wert in MLflow loggen
        mlflow.log_metric("best_val_f1", best_val_f1)

        return best_val_f1

# ============================================================
# Main
# ============================================================
def main():
    # Fix: Fallback auf "cpu" statt "gpu" (was Fehler werfen würde)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # MLflow Setup
    mlflow_path = os.path.abspath("voice_control/wake_word_detection/mlruns")
    mlflow.set_tracking_uri(f"file:///{mlflow_path}")
    mlflow.set_experiment("Wake_Word_Detection_HPO")

    # Daten laden (Test-Pfade werden geladen, aber ignoriert)
    (train_paths, train_labels), (val_paths, val_labels), _ = load_dataset_split()

    print(f"Daten geladen: Train={len(train_paths)}, Val={len(val_paths)}")
    # Hinweis: Testdaten werden hier bewusst ignoriert
    
    dataset_base_dir = "voice_control/wake_word_detection/dataset/final_dataset"
    stats_cache_path = os.path.join(dataset_base_dir, "stats.pt")
    
    batch_size = 128
    num_workers = 8  # Je nach CPU Kernen anpassen

    # Statistiken berechnen
    mean, std = compute_global_mel_stats(
        train_paths, 
        train_labels, 
        stats_cache_path,
        batch_size=batch_size,
        num_workers=num_workers
    )

    # Datasets erstellen
    train_ds = MelDataset(train_paths, train_labels, mean=mean, std=std)
    val_ds = MelDataset(val_paths, val_labels, mean=mean, std=std)
    
    # Dataloader erstellen (persistent_workers=True beschleunigt kleine Dateien)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, persistent_workers=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, persistent_workers=True)

    # Optuna Study starten
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=15, n_warmup_steps=5)
    )

    print("Starte Hyperparameter Optimierung...")
    
    # Parent Run in MLflow
    with mlflow.start_run(run_name="Optuna_Main_Study"):
        try:
            study.optimize(
                partial(objective,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        device=device,
                        epochs=35), # Epochen pro Trial
                n_trials=500,        # Anzahl der Versuche
                show_progress_bar=True
            )
        except KeyboardInterrupt:
            print("Optimierung manuell gestoppt.")

        # Beste Parameter loggen
        print("\n--- Beste Hyperparameter ---")
        print(study.best_params)
        print(f"Bester Validation F1: {study.best_value:.4f}")
        
        mlflow.log_params(study.best_params)
        mlflow.log_metric("overall_best_f1", study.best_value)

if __name__ == "__main__":
    main()