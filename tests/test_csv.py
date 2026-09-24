"""CSV target validation and positive-class semantics."""
import numpy as np
import pandas as pd
import pytest
from src.data import load_data
from src.train import main


def csv_config(tmp_path, labels, **options):
    path = tmp_path / 'data.csv'
    pd.DataFrame({'feature': np.arange(60), 'label': np.resize(labels, 60)}).to_csv(path, index=False)
    return {'seed': 42, 'data': {'source': 'csv', 'csv_path': str(path),
            'target': 'label', 'test_size': .2, **options}}


@pytest.mark.parametrize('labels,options', [([0, 1], {}),
    (['benign', 'attack'], {'positive_label': 'attack'}), ([1, 2], {'positive_label': 2}),
    ([0, 1], {'positive_label': 0})])
def test_binary_csv_mapping(tmp_path, labels, options):
    cfg = csv_config(tmp_path, labels, **options)
    Xtr, Xte, ytr, yte = load_data(cfg)
    # Reproduce the split to check positive-class semantics per row.
    from sklearn.model_selection import train_test_split
    indices = np.arange(60)
    raw = np.resize(labels, 60)
    positive = options.get('positive_label', 1)
    all_y = (raw == positive).astype('int64')
    train, test = train_test_split(indices, test_size=.2, random_state=42, stratify=all_y)
    assert np.allclose(Xtr[:, 0], (train - train.mean()) / train.std())
    assert np.allclose(Xte[:, 0], (test - train.mean()) / train.std())
    assert np.array_equal(ytr, all_y[train])
    assert np.array_equal(yte, all_y[test])
    assert set(ytr) == {0, 1}
    assert ytr.dtype == np.int64
    assert np.allclose(Xtr.mean(axis=0), 0, atol=1e-6)


@pytest.mark.parametrize('labels,options,message', [
    (['benign', 'attack'], {}, 'positive_label'),
    ([1, 2], {}, 'positive_label'),
    ([0, 1, 2], {}, 'binary classification'),
    ([1], {}, 'binary classification'),
    ([0, None], {}, 'missing labels'),
    ([0, 1], {'positive_label': 'attack'}, 'must match'),
])
def test_invalid_csv_targets(tmp_path, labels, options, message):
    with pytest.raises(ValueError, match=message):
        load_data(csv_config(tmp_path, labels, **options))


def test_string_labels_train_end_to_end(tmp_path):
    import json
    import yaml
    cfg = csv_config(tmp_path, ['benign', 'attack'], positive_label='attack')
    cfg.update(model={'hidden_sizes': [4], 'dropout': 0.0},
               train={'epochs': 1, 'batch_size': 16, 'lr': .001, 'weight_decay': 0},
               output={'dir': str(tmp_path / 'custom-results')})
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump(cfg))
    main(str(config))
    metrics = json.loads((tmp_path / 'custom-results/metrics.json').read_text())
    assert all(0 <= value <= 1 for value in metrics.values())
    for name in ['confusion_matrix.png', 'roc_curve.png', 'model.pt']:
        assert (tmp_path / 'custom-results' / name).stat().st_size > 0
