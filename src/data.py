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


def load_data(cfg: dict):
    """Return (X_train, X_test, y_train, y_test) as float32 / int64 numpy arrays."""
    d = cfg["data"]
    if d["source"] == "synthetic":
        X, y = make_classification(
            n_samples=d["n_samples"],
            n_features=d["n_features"],
            n_informative=max(2, d["n_features"] // 2),
            n_classes=2,
            random_state=cfg["seed"],
        )
    elif d["source"] == "csv":
        # Real data path: a CSV with a `target` column.
        df = pd.read_csv(d["csv_path"])
        labels = df[d["target"]]
        if labels.isna().any():
            raise ValueError("CSV target contains missing labels; clean them before training.")
        classes = labels.unique()
        if len(classes) != 2:
            raise ValueError(
                "This template supports binary classification only; "
                f"found {len(classes)} target classes. Map labels to two classes "
                "or adapt the model and evaluation for multiclass classification."
            )
        positive = d.get("positive_label", 1)
        if "positive_label" not in d and set(classes) != {0, 1}:
            raise ValueError(
                "Set data.positive_label to the CSV label to evaluate as class 1 "
                "(for example, 'attack'); the other label becomes class 0."
            )
        if positive not in classes:
            raise ValueError("data.positive_label must match one of the two CSV target labels.")
        y = (labels == positive).to_numpy(dtype="int64")
        X = df.drop(columns=[d["target"]]).select_dtypes("number").to_numpy()

    else:
        raise ValueError("data.source must be synthetic or csv.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=d["test_size"], random_state=cfg["seed"], stratify=y
    )
    # Scale using TRAIN statistics only (avoid data leakage — see L04/Unit I).
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype("float32")
    X_test = scaler.transform(X_test).astype("float32")
    return X_train, X_test, y_train.astype("int64"), y_test.astype("int64")
