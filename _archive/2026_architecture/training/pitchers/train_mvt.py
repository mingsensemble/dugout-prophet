"""
Training script for MVTPitcherModel.

MLflow experiment: pitchers/mvt
Each Optuna trial is logged as a child run. The best trial is refit and
saved as an artifact; its run is tagged stage=production.

Usage:
    python training/pitchers/train_mvt.py
    python training/pitchers/train_mvt.py --n-trials 50 --epochs 200
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
from src.pitchers.war_mvt_model import MVTPitcherDataset, MVTPitcherModel

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_PATH = "./data/pitching_stats.csv"
MODELS_DIR = "./models/pitchers/mvt/production"
TARGET = "WAR"
FEATURES = [
    "G", "GS", "IP", "TBF", "HR", "BB", "SO", "GB", "GB%", "FB", "FB%",
    "LD", "Pitches", "CSW%", "K%", "BB%", "FIP",
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


def prepare_data(X, y_features, y_war, features_list, device, batch_size=32):
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

    y_war_np = y_war.cpu().numpy()
    war_mean, war_std = float(np.mean(y_war_np)), float(np.std(y_war_np))
    y_war_norm = torch.tensor((y_war_np - war_mean) / (war_std + 1e-8), dtype=torch.float32)

    X_tr, X_val, yf_tr, yf_val, yw_tr, yw_val = train_test_split(
        X_scaled, y_norm, y_war_norm, test_size=0.2, random_state=42
    )
    train_loader = DataLoader(TensorDataset(X_tr.to(device), yf_tr.to(device), yw_tr.to(device)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val.to(device), yf_val.to(device), yw_val.to(device)), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, scaler, feature_stats, war_mean, war_std


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, device, num_epochs=100, lr=1e-3,
                patience=15, feature_weight=0.3, war_weight=0.7):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss, best_state, patience_counter = float("inf"), None, 0

    for _ in range(num_epochs):
        model.train()
        for X_b, yf_b, yw_b in train_loader:
            optimizer.zero_grad()
            fp, wp = model(X_b)
            (feature_weight * criterion(fp, yf_b) + war_weight * criterion(wp, yw_b)).backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, yf_b, yw_b in val_loader:
                fp, wp = model(X_b)
                val_loss += (feature_weight * criterion(fp, yf_b) + war_weight * criterion(wp, yw_b)).item()
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
    ds = MVTPitcherDataset(data=data, features=FEATURES, target=TARGET, nlookbacks=5, min_qual_ip=MIN_QUAL_IP)

    def objective(trial):
        d_model = trial.suggest_categorical("d_model", [32, 64, 128])
        nhead = trial.suggest_categorical("nhead", [2, 4, 8])
        if d_model % nhead != 0:
            raise optuna.TrialPruned()

        params = {
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": trial.suggest_int("num_layers", 1, 4),
            "dim_feedforward": trial.suggest_categorical("dim_feedforward", [64, 128, 256]),
            "dropout": trial.suggest_float("dropout", 0.05, 0.4),
            "head_hidden1": trial.suggest_categorical("head_hidden1", [128, 256, 512]),
            "head_hidden2": trial.suggest_categorical("head_hidden2", [64, 128, 256]),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "feature_weight": trial.suggest_float("feature_weight", 0.1, 0.5),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        }
        war_weight = 1.0 - params["feature_weight"]

        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            mlflow.log_params(params)

            train_loader, val_loader, _, _, _, _ = prepare_data(
                ds.sequences, ds.targets_features, ds.ult_target, FEATURES, device, params["batch_size"]
            )
            model = MVTPitcherModel(
                input_dim=len(FEATURES), num_features=len(FEATURES),
                d_model=params["d_model"], nhead=params["nhead"],
                num_layers=params["num_layers"], dim_feedforward=params["dim_feedforward"],
                dropout=params["dropout"], seq_len=5,
                head_hidden1=params["head_hidden1"], head_hidden2=params["head_hidden2"],
            )
            val_loss = train_model(model, train_loader, val_loader, device, num_epochs,
                                   params["lr"], patience, params["feature_weight"], war_weight)
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

    mlflow.set_experiment("pitchers/mvt")
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

        ds = MVTPitcherDataset(data=data, features=FEATURES, target=TARGET, nlookbacks=5, min_qual_ip=MIN_QUAL_IP)
        train_loader, val_loader, scaler, feature_stats, war_mean, war_std = prepare_data(
            ds.sequences, ds.targets_features, ds.ult_target, FEATURES, device, best_p["batch_size"]
        )
        best_model = MVTPitcherModel(
            input_dim=len(FEATURES), num_features=len(FEATURES),
            d_model=best_p["d_model"], nhead=best_p["nhead"],
            num_layers=best_p["num_layers"], dim_feedforward=best_p["dim_feedforward"],
            dropout=best_p["dropout"], seq_len=5,
            head_hidden1=best_p["head_hidden1"], head_hidden2=best_p["head_hidden2"],
        )
        fw = best_p["feature_weight"]
        train_model(best_model, train_loader, val_loader, device, num_epochs, best_p["lr"], patience, fw, 1.0 - fw)

        # Evaluate
        best_model.eval()
        war_preds, war_actuals = [], []
        with torch.no_grad():
            for X_b, _, yw_b in val_loader:
                _, wp = best_model(X_b)
                war_preds.append((wp.cpu().numpy() * war_std + war_mean))
                war_actuals.append((yw_b.cpu().numpy() * war_std + war_mean))
        war_preds = np.concatenate(war_preds)
        war_actuals = np.concatenate(war_actuals)

        mse = mean_squared_error(war_actuals, war_preds)
        mae = mean_absolute_error(war_actuals, war_preds)
        r2 = r2_score(war_actuals, war_preds)
        mlflow.log_metrics({"war_mse": mse, "war_mae": mae, "war_r2": r2})
        print(f"Production — WAR MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

        # Save artifacts
        os.makedirs(MODELS_DIR, exist_ok=True)
        torch.save(best_model.state_dict(), os.path.join(MODELS_DIR, "model.pt"))
        with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)

        metadata = {
            "model_type": "MVTPitcherModel",
            "hyperparameters": {k: (float(v) if isinstance(v, float) else int(v) if isinstance(v, int) else v) for k, v in best_p.items()},
            "features": FEATURES,
            "target": TARGET,
            "feature_stats": {k: list(v) for k, v in feature_stats.items()},
            "war_mean": war_mean,
            "war_std": war_std,
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
