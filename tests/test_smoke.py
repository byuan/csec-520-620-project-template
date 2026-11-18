"""Smoke tests: the pipeline runs and produces sane metrics."""
import numpy as np
from src.data import load_data, load_scaler, save_scaler
from src.evaluate import compute_metrics

CFG = {
    "seed": 0,
    "data": {"source": "synthetic", "n_samples": 400, "n_features": 10, "test_size": 0.25},
}


def test_load_data_shapes():
    Xtr, Xte, ytr, yte = load_data(CFG)
    assert Xtr.shape[1] == Xte.shape[1] == 10
    assert len(ytr) == 300 and len(yte) == 100


def test_scaler_round_trip(tmp_path):
    Xtr, Xte, ytr, yte, scaler, names = load_data(CFG, return_scaler=True)
    assert names == [f"x{i}" for i in range(10)]
    path = tmp_path / "scaler.npz"
    save_scaler(str(path), scaler, names)
    mean, scale, loaded_names = load_scaler(str(path))
    assert loaded_names == names
    # Undo the scaling with the saved values, then redo it: must match exactly.
    raw = scaler.inverse_transform(Xte)
    np.testing.assert_allclose(((raw - mean) / scale).astype("float32"), Xte, atol=1e-5)


def test_metrics_range():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.9, 0.2, 0.8])
    m = compute_metrics(y, p)
    assert 0.0 <= m["roc_auc"] <= 1.0
    assert m["accuracy"] == 1.0
