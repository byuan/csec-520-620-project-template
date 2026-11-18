"""Data loading and preparation.

Replace `load_data` with your real security dataset (e.g., CIC-IoT2023, UNSW-NB15).
The synthetic default lets `make reproduce` run end-to-end out of the box.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(cfg: dict, return_scaler: bool = False):
    """Return (X_train, X_test, y_train, y_test) as float32 / int64 numpy arrays.

    With return_scaler=True, also return the fitted StandardScaler and the list of
    feature names, in column order: (X_train, X_test, y_train, y_test, scaler, names).
    """
    d = cfg["data"]
    if d["source"] == "synthetic":
        X, y = make_classification(
            n_samples=d["n_samples"],
            n_features=d["n_features"],
            n_informative=max(2, d["n_features"] // 2),
            n_classes=2,
            random_state=cfg["seed"],
        )
        feature_names = [f"x{i}" for i in range(X.shape[1])]
    else:
        # Real data path: a CSV with a `target` column.
        df = pd.read_csv(d["csv_path"])
        y = df[d["target"]].to_numpy()
        features = df.drop(columns=[d["target"]]).select_dtypes("number")
        feature_names = list(features.columns)
        X = features.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=d["test_size"], random_state=cfg["seed"], stratify=y
    )
    # Scale using TRAIN statistics only (avoid data leakage — see L04/Unit I).
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype("float32")
    X_test = scaler.transform(X_test).astype("float32")
    out = (X_train, X_test, y_train.astype("int64"), y_test.astype("int64"))
    if return_scaler:
        return out + (scaler, feature_names)
    return out


def save_scaler(path: str, scaler: StandardScaler, feature_names: list[str]) -> None:
    """Save the fitted scaler as plain arrays (no pickle), so the model can be reused.

    The trained model only works on inputs scaled exactly as in training, so keep
    this file with model.pt (e.g., to deploy the model or attack it later).
    """
    np.savez(path,
             mean=scaler.mean_.astype("float64"),
             scale=scaler.scale_.astype("float64"),
             feature_names=np.array(feature_names, dtype=str))


def load_scaler(path: str):
    """Load a scaler saved by save_scaler. Returns (mean, scale, feature_names).

    Scale new raw data the same way as in training with: (X_raw - mean) / scale
    """
    with np.load(path, allow_pickle=False) as f:
        return f["mean"], f["scale"], [str(n) for n in f["feature_names"]]
