import os
from functools import partial
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import mlflow
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from models import CRNNWakeWord


# ============================================================
# Dataset-Klasse
# ============================================================
class MelDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        mel_spec = torch.load(self.file_paths[idx], weights_only=True)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return mel_spec, label


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
        dataset_split.append((paths, labels))
    return tuple(dataset_split)


# ============================================================
# Training / Validation / Evaluation
# ============================================================
def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

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

        running_loss += loss.item() * X.size(0)

    return running_loss / len(dataloader.dataset)


def validate_fast(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            running_loss += loss.item() * X.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return running_loss / total, correct / total


def evaluate_full(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            preds = model(X).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    acc = (all_preds == all_labels).mean()
    precision = precision_score(all_labels, all_preds, average="binary")
    recall = recall_score(all_labels, all_preds, average="binary")
    f1 = f1_score(all_labels, all_preds, average="binary")
    conf_matrix = confusion_matrix(all_labels, all_preds)
    return acc, precision, recall, f1, conf_matrix


# ============================================================
# Optuna Objective mit MLflow Logging
# ============================================================
def objective(trial, train_loader, val_loader, test_loader, device, epochs=15):
    # ----------------- Hyperparameter -----------------
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64])
    num_layers = trial.suggest_int("num_layers", 1, 2)
    conv_channels = trial.suggest_categorical("conv_channels", [32, 64, 128])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

    # ----------------- Modell & Optimierung -----------------
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
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    # ----------------- MLflow Trial Logging -----------------
    with mlflow.start_run(run_name=f"Trial_{trial.number}", nested=True):
        mlflow.log_params({
            "lr": lr,
            "dropout": dropout,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "conv_channels": conv_channels,
            "weight_decay": weight_decay,
            "total_params": sum(p.numel() for p in model.parameters())
        })

        best_val_acc = 0.0
        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
            val_loss, val_acc = validate_fast(model, val_loader, criterion, device)
            scheduler.step(val_loss)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)
            mlflow.log_metric("lr", optimizer.param_groups[0]['lr'], step=epoch)

            print(f"[Trial {trial.number}] Epoch {epoch+1}/{epochs}: "
                  f"Train_Loss={train_loss:.4f} | Val_Loss={val_loss:.4f} | Val_Acc={val_acc:.4f}")

            trial.report(val_acc, epoch)
            if trial.should_prune():
                mlflow.log_param("pruned", True)
                raise optuna.TrialPruned()

        # ----------------- Nach Training: Test-Evaluation -----------------
        acc, precision, recall, f1, conf_matrix = evaluate_full(model, test_loader, device)
        print(f"\n[Trial {trial.number}] ✅ Test-F1={f1:.4f}, Acc={acc:.4f}")
        print(f"Confusion Matrix:\n{conf_matrix}")

        mlflow.log_metrics({
            "test_acc": acc,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1
        })

        # Confusion Matrix speichern
        cm_path = f"conf_matrix_trial_{trial.number}.npy"
        np.save(cm_path, conf_matrix)
        mlflow.log_artifact(cm_path)
        os.remove(cm_path)

        return f1


# ============================================================
# Main
# ============================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Using device: {device}")

    # Pfade
    mlflow_tracking_path = os.path.abspath("voice_control/wake_word_detection/mlruns")
    mlflow.set_tracking_uri(f"file:///{mlflow_tracking_path}")
    mlflow.set_experiment("wake_word_detection_CRNN")

    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = load_dataset_split()
    print(f"📊 Dataset sizes: Train={len(train_paths)}, Val={len(val_paths)}, Test={len(test_paths)}")

    # DataLoader
    batch_size = 1024
    num_workers = 8
    train_loader = DataLoader(MelDataset(train_paths, train_labels), batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, prefetch_factor=4)
    val_loader = DataLoader(MelDataset(val_paths, val_labels), batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, prefetch_factor=4)
    test_loader = DataLoader(MelDataset(test_paths, test_labels), batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True, prefetch_factor=4)

    # Optuna Study
    n_trials, epochs = 3, 15
    study = optuna.create_study(direction="maximize",
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=3))

    # MLflow-Hauptlauf
    with mlflow.start_run(run_name="Optuna_Study"):
        study.optimize(
            partial(objective,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    device=device,
                    epochs=epochs),
            n_trials=n_trials,
            show_progress_bar=True
        )

        # Beste Ergebnisse loggen
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_value", study.best_value)

    print("\n🏆 BEST HYPERPARAMETERS:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
