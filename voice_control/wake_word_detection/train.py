import os
from datetime import datetime
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import mlflow

from dataset.dataset import get_dataloaders
from models import CRNNWakeWord

# -------------------------------
# Training and Validation
# -------------------------------
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
    correct = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            running_loss += loss.item() * X.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()

    return running_loss / len(dataloader.dataset), correct / len(dataloader.dataset)


# -------------------------------
# Optuna Objective
# -------------------------------
def objective(trial, train_loader, val_loader, device, save_dir, epochs=15):
    # Hyperparameter Search
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    conv_channels = trial.suggest_categorical("conv_channels", [32, 64, 128])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

    # CRNN Modell
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

    # Log Hyperparameter in MLflow
    total_params = sum(p.numel() for p in model.parameters())
    mlflow.start_run(nested=True)
    mlflow.log_params({
        "model": "CRNN",
        "lr": lr,
        "dropout": dropout,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "conv_channels": conv_channels,
        "weight_decay": weight_decay,
        "total_params": total_params
    })

    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr
        }, step=epoch)

        print(f"  [Trial {trial.number}] Epoch {epoch+1}/{epochs}: "
              f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
              f"Val Acc={val_acc:.4f}, LR={current_lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            model_path = os.path.join(save_dir, f"best_CRNN_trial_{trial.number}.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'hyperparameters': {
                    'conv_channels': conv_channels,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'dropout': dropout
                }
            }, model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        trial.report(val_loss, epoch)
        if trial.should_prune():
            mlflow.end_run()
            raise optuna.TrialPruned()

    mlflow.log_metric("best_val_acc", best_val_acc)
    mlflow.log_metric("best_val_loss", best_val_loss)
    mlflow.end_run()

    return best_val_loss


# -------------------------------
# Test Function
# -------------------------------
def test_model(model, test_loader, criterion, device):
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"\n{'='*50}")
    print(f"📊 FINAL TEST RESULTS")
    print(f"{'='*50}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"{'='*50}\n")
    return test_loss, test_acc


# -------------------------------
# Main
# -------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Using device: {device}")

    save_dir = "voice_control/wake_word_detection/checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    n_trials = 20
    epochs = 15
    batch_size = 32
    num_workers = 2

    mlflow_tracking_path = "voice_control/wake_word_detection/mlruns"
    mlflow.set_tracking_uri(mlflow_tracking_path)
    mlflow.set_experiment("wake_word_detection_CRNN")
    print(f"📊 MLflow tracking path: {mlflow_tracking_path}")

    # Load Dataset (nur CRNN)
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers
    )

    print(f"📊 Dataset sizes:")
    print(f"  Train: {len(train_loader.dataset)}")
    print(f"  Val: {len(val_loader.dataset)}")
    print(f"  Test: {len(test_loader.dataset)}")

    # Optuna Study
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    print(f"\n🔍 Starting hyperparameter search ({n_trials} trials)...")
    study.optimize(
        partial(objective, train_loader=train_loader,
                val_loader=val_loader, device=device,
                save_dir=save_dir, epochs=epochs),
        n_trials=n_trials,
        show_progress_bar=True
    )

    print(f"\n🏆 BEST HYPERPARAMETERS:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"  Best validation loss: {study.best_value:.4f}")

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

    test_loss, test_acc = test_model(best_model, test_loader, criterion, device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = os.path.join(save_dir, f"CRNN_final_{timestamp}.pt")
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'hyperparameters': checkpoint['hyperparameters'],
        'test_acc': test_acc,
        'test_loss': test_loss
    }, final_model_path)
    print(f"💾 Final model saved to: {final_model_path}")


if __name__ == "__main__":
    main()
