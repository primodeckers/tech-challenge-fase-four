import json

import numpy as np
import pandas as pd
import pytest

from credit_monitor.audit import audit_groups
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
