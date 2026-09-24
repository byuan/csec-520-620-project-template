"""Grading must validate required metrics at the configured location."""
import json
import pytest
from grading import grade


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setattr(grade, 'ROOT', tmp_path)
    (tmp_path / 'config.yaml').write_text('output:\n  dir: custom-results\n')
    (tmp_path / 'custom-results').mkdir()
    return tmp_path


def write_metrics(repo, values):
    (repo / 'custom-results/metrics.json').write_text(json.dumps(values))


def test_custom_output_ignores_stale_default(repo):
    (repo / 'results').mkdir()
    (repo / 'results/metrics.json').write_text('{}')
    write_metrics(repo, {k: 0.9 for k in grade.METRIC_KEYS})
    assert grade.check_metrics()['status'] == 'pass'


def test_missing_custom_output_does_not_use_default(repo):
    (repo / 'results').mkdir()
    (repo / 'results/metrics.json').write_text(json.dumps({k: .9 for k in grade.METRIC_KEYS}))
    assert grade.check_metrics()['status'] == 'fail'


@pytest.mark.parametrize('value', [None, '0.9', True, False, [], {}, float('nan'),
                                  float('inf'), -float('inf'), -0.1, 1.1])
def test_invalid_metric_values(repo, value):
    metrics = {k: .9 for k in grade.METRIC_KEYS}
    metrics['f1'] = value
    write_metrics(repo, metrics)
    assert grade.check_metrics()['status'] == 'fail'


@pytest.mark.parametrize('values', [[], None, 'invalid', 1, {}, {'accuracy': .9}])
def test_invalid_shape_or_missing_keys(repo, values):
    write_metrics(repo, values)
    assert grade.check_metrics()['status'] == 'fail'


def test_valid_boundaries_and_metadata(repo):
    values = dict(zip(sorted(grade.METRIC_KEYS), [0, 1, 0.0, 1.0, .5]))
    values['dataset'] = 'example'
    write_metrics(repo, values)
    assert grade.check_metrics()['status'] == 'pass'


@pytest.mark.parametrize('config', ['', 'output: null', 'output: {dir: null}',
                                   'output: {dir: ""}', 'output: ['])
def test_bad_output_configuration(repo, config):
    (repo / 'config.yaml').write_text(config)
    assert grade.check_metrics()['status'] == 'fail'


def test_absolute_output_path(repo):
    import yaml
    (repo / 'config.yaml').write_text(yaml.safe_dump({'output': {'dir': str(repo / 'custom-results')}}))
    write_metrics(repo, {k: .9 for k in grade.METRIC_KEYS})
    assert grade.check_metrics()['status'] == 'pass'


def test_malformed_metrics_json(repo):
    (repo / 'custom-results/metrics.json').write_text('{')
    assert grade.check_metrics()['status'] == 'fail'
