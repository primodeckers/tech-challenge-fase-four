import json

import numpy as np
import pandas as pd
import pytest

from credit_monitor.audit import audit_groups, run_audit
from credit_monitor.contracts import training_schema
from credit_monitor.data import TARGET


@pytest.fixture
def production():
    rows = 120
    frame = pd.DataFrame({name: np.zeros(rows, dtype='int64') for name in training_schema().columns})
    frame['ID'] = np.arange(1000, 1000 + rows)
    frame['LIMIT_BAL'] = 20000
    frame['SEX'] = np.repeat([1, 2], 60)
    frame['AGE'] = np.repeat([29, 30], 60)
    frame[TARGET] = np.tile([0, 1], 60)
    return frame


def test_group_audit_uses_one_threshold_and_reports_conditional_rates(production):
    scores = np.r_[np.full(20, .27), np.full(40, .1), np.full(60, .9)]
    result = audit_groups(production, scores, .27)
    first = result['groups']['SEX']['1']
    second = result['groups']['SEX']['2']
    assert first['rows'] == 60 and first['observed_positive_count'] == 30
    assert first['observed_negative_count'] == 30
    assert first['predicted_positive_rate'] == pytest.approx(1 / 3)
    assert first['recall'] == pytest.approx(1 / 3)
    assert first['false_positive_rate'] == pytest.approx(1 / 3)
    assert second['recall'] == second['false_positive_rate'] == 1
    assert result['gaps_percentage_points']['SEX']['recall'] == pytest.approx(200 / 3)
    assert result['groups']['AGE']['19-29'] == first
    assert result['groups']['AGE']['30-44'] == second
    assert 'SEX' not in result['model_features'] and 'AGE' in result['model_features']
    assert 'ID' not in json.dumps(result)


def test_group_audit_marks_small_groups_and_low_denominators(production):
    batch = production.iloc[:70].copy()
    batch[TARGET] = 0
    batch.loc[0:3, TARGET] = 1
    result = audit_groups(batch, np.full(len(batch), .1), .27)
    group = result['groups']['SEX']['1']
    assert group['observed_positive_count'] == 4
    assert group['recall'] is None and group['false_negative_rate'] is None
    assert group['false_positive_rate'] == 0
    assert result['groups']['SEX']['2'] == {'status': 'suppressed_small_group'}
    assert result['groups']['AGE']['60-100'] == {'status': 'suppressed_small_group'}
    assert result['gaps_percentage_points']['SEX']['false_positive_rate'] is None


@pytest.mark.parametrize('scores', [[.4], np.full(120, float('nan')), np.full(120, 1.01)])
def test_group_audit_rejects_invalid_or_unpaired_probabilities(production, scores):
    with pytest.raises(ValueError, match='Probabilidades'):
        audit_groups(production, scores, .27)


def test_file_audit_preserves_model_threshold_lineage_and_previous_report(inputs, tmp_path):
    destination = tmp_path / 'audit'
    report = run_audit(inputs['data_dir'], inputs['model_dir'], destination, trusted_model=True)
    serialized = (destination / 'audit.json').read_bytes()
    assert json.loads(serialized) == report
    assert report['rows'] == 240 and report['threshold'] == .27
    assert report['groups']['SEX']['1']['rows'] == 120
    assert report['groups']['SEX']['2']['rows'] == 120
    model = json.loads((inputs['model_dir'] / 'manifest.json').read_text())
    assert report['model_sha256'] == model['model_sha256']
    assert report['data_manifest_sha256'] == model['metadata']['data_manifest_sha256']
    with pytest.raises(FileExistsError):
        run_audit(inputs['data_dir'], inputs['model_dir'], destination, trusted_model=True)
    assert (destination / 'audit.json').read_bytes() == serialized


@pytest.mark.parametrize('corruption', ['manifest', 'production'])
def test_file_audit_refuses_changed_lineage_without_publishing_report(inputs, tmp_path, corruption):
    name = 'manifest.json' if corruption == 'manifest' else 'production.csv'
    path = inputs['data_dir'] / name
    path.write_bytes(path.read_bytes() + b' ')
    destination = tmp_path / 'audit'
    with pytest.raises(ValueError, match='Origem|SHA-256'):
        run_audit(inputs['data_dir'], inputs['model_dir'], destination, trusted_model=True)
    assert not destination.exists()
