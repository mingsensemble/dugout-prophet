"""
Training script for CNNPitcherModel.

MLflow experiment: pitchers/cnn
Each Optuna trial is logged as a child run. The best trial is refit and
saved as an artifact; its run is tagged stage=production.

Usage:
    python training/pitchers/train_cnn.py
    python training/pitchers/train_cnn.py --n-trials 50 --epochs 200
"""
import argparse
import json
import os
import pickle
import sys

import mlflow
import mlflow.pytorch
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.pitchers.war_cnn_model import CNNPitcherDataset, CNNPitcherModel

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_PATH = "./data/pitching_stats.csv"
MODELS_DIR = "./models/pitchers/cnn/production"
TARGET = "WAR"
FEATURES = [
    "Age", "ERA", "G", "GS", "IP", "TBF", "HR", "BB", "IBB", "HBP", "SO",
    "GB", "GB%", "FB", "FB%", "LD", "LD%", "IFFB", "Pitches", "BABIP", "WHIP",
    "FIP", "xFIP", "SIERA", "CStr%", "CSW%", "Barrels", "Barrel%", "HardHit", "HardHit%",
]
MIN_QUAL_IP = 50


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_data():
    if os.path.exists(DATA_PATH):
        data = pd.read_csv(DATA_PATH)
        if "IP" in data.columns:
            print(f"Loaded {DATA_PATH} ({len(data)} rows)")
            return data
    print("Downloading pitching stats via pybaseball...")
    from pybaseball import pitching_stats
    data = pitching_stats(start_season=2015, end_season=2025, qual=10)
    data.to_csv(DATA_PATH, index=False)
    return data


def prepare_data(X, y, device, batch_size=32):
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    shape = X_tr.shape
    X_tr_scaled = torch.tensor(scaler.fit_transform(X_tr.view(shape[0], -1).numpy()).reshape(shape), dtype=torch.float32)
    X_val_scaled = torch.tensor(scaler.transform(X_val.view(X_val.shape[0], -1).numpy()).reshape(X_val.shape), dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_tr_scaled.to(device), y_tr.to(device)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_scaled.to(device), y_val.to(device)), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, scaler


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, device, num_epochs=100, lr=1e-3, patience=15):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss, best_state, patience_counter = float("inf"), None, 0

    for _ in range(num_epochs):
        model.train()
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            criterion(model(X_b).squeeze(), y_b).backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                val_loss += criterion(model(X_b).squeeze(), y_b).item() * X_b.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return best_val_loss


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------
def make_objective(data, device, num_epochs, patience):
    def objective(trial):
        params = {
            "conv1_out": trial.suggest_categorical("conv1_out", [16, 32, 64]),
            "conv1_kernel": trial.suggest_int("conv1_kernel", 1, 3),
            "conv2_out": trial.suggest_categorical("conv2_out", [32, 64, 128]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "fc1_hidden": trial.suggest_categorical("fc1_hidden", [64, 128, 256]),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "nlookbacks": trial.suggest_int("nlookbacks", 3, 6),
        }

        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            mlflow.log_params(params)

            ds = CNNPitcherDataset(data=data, features=FEATURES, target=TARGET, nlookbacks=params["nlookbacks"], min_qual_ip=MIN_QUAL_IP)
            X = torch.tensor(ds.sequences, dtype=torch.float32)
            y = torch.tensor(ds.target, dtype=torch.float32)
            train_loader, val_loader, _ = prepare_data(X, y, device, params["batch_size"])

            model = CNNPitcherModel(
                input_channels=len(FEATURES), seq_length=params["nlookbacks"],
                conv1_out=params["conv1_out"], conv1_kernel=params["conv1_kernel"],
                conv2_out=params["conv2_out"], conv2_kernel=1,
                dropout=params["dropout"], fc1_hidden=params["fc1_hidden"],
            )
            val_loss = train_model(model, train_loader, val_loader, device, num_epochs, params["lr"], patience)
            mlflow.log_metric("val_loss", val_loss)

        return val_loss

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(n_trials=10, num_epochs=100, patience=15):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    data = load_data()

    mlflow.set_experiment("pitchers/cnn")
    with mlflow.start_run(run_name="optuna_search"):
        mlflow.log_params({"n_trials": n_trials, "num_epochs": num_epochs, "patience": patience})

        study = optuna.create_study(direction="minimize")
        study.optimize(make_objective(data, device, num_epochs, patience), n_trials=n_trials)

        best_p = study.best_params
        print(f"Best val loss: {study.best_value:.6f}")
        mlflow.log_metric("best_val_loss", study.best_value)
        mlflow.log_params({f"best_{k}": v for k, v in best_p.items()})

    # ---- Retrain best configuration and log as production run ----
    with mlflow.start_run(run_name="production"):
        mlflow.log_params(best_p)
        mlflow.set_tag("stage", "production")

        ds = CNNPitcherDataset(data=data, features=FEATURES, target=TARGET, nlookbacks=best_p["nlookbacks"], min_qual_ip=MIN_QUAL_IP)
        X = torch.tensor(ds.sequences, dtype=torch.float32)
        y = torch.tensor(ds.target, dtype=torch.float32)
        train_loader, val_loader, scaler = prepare_data(X, y, device, best_p["batch_size"])

        best_model = CNNPitcherModel(
            input_channels=len(FEATURES), seq_length=best_p["nlookbacks"],
            conv1_out=best_p["conv1_out"], conv1_kernel=best_p["conv1_kernel"],
            conv2_out=best_p["conv2_out"], conv2_kernel=1,
            dropout=best_p["dropout"], fc1_hidden=best_p["fc1_hidden"],
        )
        train_model(best_model, train_loader, val_loader, device, num_epochs, best_p["lr"], patience)

        # Evaluate
        best_model.eval()
        preds, actuals = [], []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                preds.append(best_model(X_b).squeeze().cpu().numpy())
                actuals.append(y_b.cpu().numpy())
        preds = np.concatenate(preds)
        actuals = np.concatenate(actuals)

        mse = mean_squared_error(actuals, preds)
        mae = mean_absolute_error(actuals, preds)
        r2 = r2_score(actuals, preds)
        mlflow.log_metrics({"war_mse": mse, "war_mae": mae, "war_r2": r2})
        print(f"Production — WAR MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

        # Save artifacts
        os.makedirs(MODELS_DIR, exist_ok=True)
        torch.save(best_model.state_dict(), os.path.join(MODELS_DIR, "model.pt"))
        with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)

        metadata = {
            "model_type": "CNNPitcherModel",
            "hyperparameters": {k: (float(v) if isinstance(v, float) else int(v) if isinstance(v, int) else v) for k, v in best_p.items()},
            "features": FEATURES,
            "target": TARGET,
            "metrics": {"mse": float(mse), "mae": float(mae), "r2": float(r2)},
        }
        meta_path = os.path.join(MODELS_DIR, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        mlflow.log_artifacts(MODELS_DIR, artifact_path="model")
        print(f"Saved artifacts to {MODELS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()
    main(n_trials=args.n_trials, num_epochs=args.epochs, patience=args.patience)
