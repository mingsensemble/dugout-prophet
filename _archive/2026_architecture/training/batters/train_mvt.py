"""
Training script for MVTBatterModel.

MLflow experiment: batters/mvt
Each Optuna trial is logged as a child run. The best trial is refit and
saved as an artifact; its run is tagged stage=production.

Usage:
    python training/batters/train_mvt.py
    python training/batters/train_mvt.py --n-trials 50 --epochs 200
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

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.batters.mvt_model import MVTBatterDataset, MVTBatterModel

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_PATH = "./data/batting_stats.csv"
MODELS_DIR = "./models/batters/mvt/production"

SCORE_MAP = {
    "R": 0.75, "1B": 1, "2B": 1.5, "3B": 2, "HR": 3, "RBI": 0.75,
    "BB": 1, "SO": -0.5, "HBP": 1, "SB": 1, "CS": -2, "GDP": -2,
}
AUX_FEATURES = [
    "PA", "AB", "xwOBA", "OPS", "ISO", "OBP",
    "O-Swing%", "Z-Swing%", "Swing%", "O-Contact%", "Z-Contact%",
    "Contact%", "Zone%", "SwStr%", "BsR",
    "LD%", "GB%", "FB%", "HR/FB", "Hard%", "GB/FB", "Barrel%",
]
FEATURES = list(SCORE_MAP.keys()) + AUX_FEATURES
MIN_QUAL_PA = 50


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_data():
    if os.path.exists(DATA_PATH):
        data = pd.read_csv(DATA_PATH)
        if "PA" in data.columns:
            print(f"Loaded {DATA_PATH} ({len(data)} rows)")
            return data
    print("Downloading batting stats via pybaseball...")
    from pybaseball import batting_stats
    data = batting_stats(start_season=2015, end_season=2025, qual=10)
    data.to_csv(DATA_PATH, index=False)
    return data


def prepare_data(X, y_features, features_list, device, batch_size=32):
    X_np = X.cpu().numpy()
    n, seq_len, n_feat = X_np.shape
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_np.reshape(-1, n_feat)).reshape(n, seq_len, n_feat)
    X_scaled = torch.tensor(X_scaled, dtype=torch.float32)

    y_np = y_features.cpu().numpy()
    feature_stats = {}
    y_norm = np.zeros_like(y_np)
    for i, name in enumerate(features_list):
        mean, std = float(np.mean(y_np[:, i])), float(np.std(y_np[:, i]))
        feature_stats[name] = (mean, std)
        y_norm[:, i] = (y_np[:, i] - mean) / (std + 1e-8)
    y_norm = torch.tensor(y_norm, dtype=torch.float32)

    X_tr, X_val, y_tr, y_val = train_test_split(X_scaled, y_norm, test_size=0.2, random_state=42)
    train_loader = DataLoader(TensorDataset(X_tr.to(device), y_tr.to(device)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val.to(device), y_val.to(device)), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, scaler, feature_stats


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, device, num_epochs=100, lr=1e-3, patience=15):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss, best_state, patience_counter = float("inf"), None, 0

    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            criterion(model(X_batch), y_batch).backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                val_loss += criterion(model(X_batch), y_batch).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

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
        d_model = trial.suggest_categorical("d_model", [32, 64, 128])
        nhead = trial.suggest_categorical("nhead", [2, 4, 8])
        if d_model % nhead != 0:
            raise optuna.TrialPruned()

        params = {
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": trial.suggest_int("num_layers", 1, 4),
            "dim_feedforward": trial.suggest_categorical("dim_feedforward", [64, 128, 256, 512]),
            "dropout": trial.suggest_float("dropout", 0.05, 0.4),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "nlookbacks": trial.suggest_int("nlookbacks", 3, 8),
        }

        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            mlflow.log_params(params)

            ds = MVTBatterDataset(data=data, features=FEATURES, nlookbacks=params["nlookbacks"], min_qual_pa=MIN_QUAL_PA)
            train_loader, val_loader, _, _ = prepare_data(ds.sequences, ds.targets_features, FEATURES, device, params["batch_size"])

            model = MVTBatterModel(
                input_dim=len(FEATURES), num_features=len(FEATURES),
                d_model=params["d_model"], nhead=params["nhead"],
                num_layers=params["num_layers"], dim_feedforward=params["dim_feedforward"],
                dropout=params["dropout"], nlookbacks=params["nlookbacks"],
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

    mlflow.set_experiment("batters/mvt")
    with mlflow.start_run(run_name="optuna_search"):
        mlflow.log_params({"n_trials": n_trials, "num_epochs": num_epochs, "patience": patience})

        study = optuna.create_study(direction="minimize")
        study.optimize(make_objective(data, device, num_epochs, patience), n_trials=n_trials)

        best_p = study.best_params
        print(f"Best val loss: {study.best_value:.6f}")
        print(f"Best params: {best_p}")
        mlflow.log_metric("best_val_loss", study.best_value)
        mlflow.log_params({f"best_{k}": v for k, v in best_p.items()})

    # ---- Retrain best configuration and log as production run ----
    with mlflow.start_run(run_name="production"):
        mlflow.log_params(best_p)
        mlflow.set_tag("stage", "production")

        ds = MVTBatterDataset(data=data, features=FEATURES, nlookbacks=best_p["nlookbacks"], min_qual_pa=MIN_QUAL_PA)
        train_loader, val_loader, scaler, feature_stats = prepare_data(
            ds.sequences, ds.targets_features, FEATURES, device, best_p["batch_size"]
        )

        best_model = MVTBatterModel(
            input_dim=len(FEATURES), num_features=len(FEATURES),
            d_model=best_p["d_model"], nhead=best_p["nhead"],
            num_layers=best_p["num_layers"], dim_feedforward=best_p["dim_feedforward"],
            dropout=best_p["dropout"], nlookbacks=best_p["nlookbacks"],
        )
        train_model(best_model, train_loader, val_loader, device, num_epochs, best_p["lr"], patience)

        # Evaluate
        best_model.eval()
        all_preds, all_actuals = [], []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                all_preds.append(best_model(X_b).cpu().numpy())
                all_actuals.append(y_b.cpu().numpy())
        all_preds = np.concatenate(all_preds)
        all_actuals = np.concatenate(all_actuals)

        preds_d, actuals_d = np.zeros_like(all_preds), np.zeros_like(all_actuals)
        for i, fname in enumerate(FEATURES):
            mean, std = feature_stats[fname]
            preds_d[:, i] = all_preds[:, i] * std + mean
            actuals_d[:, i] = all_actuals[:, i] * std + mean

        overall_mse = mean_squared_error(all_actuals, all_preds)
        overall_mae = mean_absolute_error(all_actuals, all_preds)
        overall_r2 = r2_score(all_actuals.flatten(), all_preds.flatten())
        mlflow.log_metrics({"val_mse": overall_mse, "val_mae": overall_mae, "val_r2": overall_r2})
        print(f"Production — MSE: {overall_mse:.4f}, MAE: {overall_mae:.4f}, R²: {overall_r2:.4f}")

        # Save artifacts
        os.makedirs(MODELS_DIR, exist_ok=True)
        torch.save(best_model.state_dict(), os.path.join(MODELS_DIR, "model.pt"))
        with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)

        metadata = {
            "model_type": "MVTBatterModel",
            "hyperparameters": {k: (float(v) if isinstance(v, float) else int(v) if isinstance(v, int) else v) for k, v in best_p.items()},
            "features": FEATURES,
            "score_map": SCORE_MAP,
            "feature_stats": {k: list(v) for k, v in feature_stats.items()},
            "metrics": {"mse": float(overall_mse), "mae": float(overall_mae), "r2": float(overall_r2)},
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
