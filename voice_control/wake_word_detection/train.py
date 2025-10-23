import os
from datetime import datetime
from functools import partial
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import mlflow
from torch.utils.data import DataLoader

from dataset.dataset import WakeWordDataset, load_dataset_split
from models import CRNNWakeWord
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# ===============================
# Training and validation
# ===============================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X.size(0)
    return running_loss / len(dataloader.dataset)

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            running_loss += loss.item() * X.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = (all_preds == all_labels).mean()
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    val_loss = running_loss / len(dataloader.dataset)
    return val_loss, acc, precision, recall, f1, conf_matrix

# ===============================
# Optuna objective function
# ===============================
def objective(trial, train_loader, val_loader, device, save_dir, epochs=15):
    # Hyperparameter suggestions
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64])
    num_layers = trial.suggest_int("num_layers", 1, 2)
    conv_channels = trial.suggest_categorical("conv_channels", [32, 64, 128])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

    # Model
    model = CRNNWakeWord(
        num_classes=2,
        conv_channels=conv_channels,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # MLflow run
    run_name = f"Trial_{trial.number}"
    mlflow.start_run(nested=True, run_name=run_name)
    mlflow.log_params({
        "lr": lr,
        "dropout": dropout,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "conv_channels": conv_channels,
        "weight_decay": weight_decay,
        "total_params": sum(p.numel() for p in model.parameters())
    })

    best_val_f1 = 0.0
    best_model_path = os.path.join(save_dir, f"best_CRNN_trial_{trial.number}.pt")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_prec, val_rec, val_f1, val_conf = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        # MLflow metrics
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_precision": val_prec,
            "val_recall": val_rec,
            "val_f1": val_f1,
            "lr": optimizer.param_groups[0]['lr']
        }, step=epoch)

        print(f"[Trial {trial.number}] Epoch {epoch+1}/{epochs}: "
              f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
              f"Acc={val_acc:.4f}, Prec={val_prec:.4f}, Rec={val_rec:.4f}, F1={val_f1:.4f}")
        print(f"Confusion Matrix:\n{val_conf}")

        # Save best model based on F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                'model_state_dict': model.state_dict(),
                'hyperparameters': {
                    'conv_channels': conv_channels,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'dropout': dropout
                }
            }, best_model_path)

        # Optuna pruning
        trial.report(val_f1, epoch)
        if trial.should_prune():
            mlflow.end_run()
            raise optuna.TrialPruned()

    mlflow.end_run()
    return best_val_f1

# ===============================
# Test function
# ===============================
def test_model(model, test_loader, criterion, device):
    test_loss, test_acc, test_prec, test_rec, test_f1, test_conf = validate(model, test_loader, criterion, device)

    print(f"\n{'='*50}")
    print(f"📊 FINAL TEST RESULTS")
    print(f"{'='*50}")
    print(f"Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, "
          f"Prec: {test_prec:.4f}, Rec: {test_rec:.4f}, F1: {test_f1:.4f}")
    print(f"Confusion Matrix:\n{test_conf}")
    print(f"{'='*50}\n")

    return test_loss, test_acc, test_prec, test_rec, test_f1, test_conf

# ===============================
# Main function
# ===============================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Using device: {device}")

    save_dir = "voice_control/wake_word_detection/checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    n_trials = 50
    epochs = 30
    batch_size = 32
    num_workers = 4

    # MLflow
    mlflow_tracking_path = "voice_control/wake_word_detection/mlruns"
    mlflow.set_tracking_uri(mlflow_tracking_path)
    mlflow.set_experiment("wake_word_detection_CRNN")
    print(f"📊 MLflow tracking path: {mlflow_tracking_path}")

    # Load dataset splits
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = load_dataset_split()

    train_dataset = WakeWordDataset(train_paths, train_labels)
    val_dataset = WakeWordDataset(val_paths, val_labels)
    test_dataset = WakeWordDataset(test_paths, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"📊 Dataset sizes: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")

    # Optuna study: maximize F1
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5))
    print(f"\n🔍 Starting hyperparameter search ({n_trials} trials)...")
    study.optimize(
        partial(objective, train_loader=train_loader, val_loader=val_loader, device=device, save_dir=save_dir, epochs=epochs),
        n_trials=n_trials,
        show_progress_bar=True
    )

    print(f"\n🏆 BEST HYPERPARAMETERS:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"  Best validation F1: {study.best_value:.4f}")

    # Load best model for testing
    best_trial = study.best_trial
    model_path = os.path.join(save_dir, f"best_CRNN_trial_{best_trial.number}.pt")
    checkpoint = torch.load(model_path)

    best_model = CRNNWakeWord(
        num_classes=2,
        **checkpoint['hyperparameters']
    ).to(device)
    best_model.load_state_dict(checkpoint['model_state_dict'])
    criterion = nn.CrossEntropyLoss()

    test_model(best_model, test_loader, criterion, device)

if __name__ == "__main__":
    main()
