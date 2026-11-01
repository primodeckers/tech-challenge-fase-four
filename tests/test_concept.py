import numpy as np
import pandas as pd
import pytest

from credit_monitor.contracts import InvalidBatch
from credit_monitor.data import TARGET
from credit_monitor.scenarios import build_concept_scenario


@pytest.fixture
def production():
    n = 32
    values = {"ID": np.arange(1001, 1001+n), "LIMIT_BAL": np.full(n, 100000),
              "SEX": np.tile([1, 2], n//2), "EDUCATION": np.ones(n),
              "MARRIAGE": np.ones(n), "AGE": np.full(n, 35)}
    for col in ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"):
        values[col] = np.repeat([-1, 2], n//2)
    for i in range(1, 7):
        values[f"BILL_AMT{i}"] = np.full(n, 1000)
        values[f"PAY_AMT{i}"] = np.full(n, 100)
    values[TARGET] = np.repeat([0, 1], n//2)
    return pd.DataFrame(values, dtype="int64")


def test_concept_preserves_x_source_and_prevalence_with_exact_conditional_swaps(production):
    original = production.copy(deep=True)
    changed = build_concept_scenario(production)
    pd.testing.assert_frame_equal(changed.drop(columns=TARGET), original.drop(columns=TARGET))
    pd.testing.assert_frame_equal(production, original)
    up = (original[TARGET] == 0) & (changed[TARGET] == 1)
    down = (original[TARGET] == 1) & (changed[TARGET] == 0)
    assert up.sum() == down.sum() == 4
    assert (production.loc[up, 'PAY_0'] <= 0).all()
    assert (production.loc[down, 'PAY_0'] > 0).all()
    assert changed[TARGET].sum() == original[TARGET].sum()
    assert all(dtype == np.dtype('int64') for dtype in changed.dtypes)
    changed.loc[0, 'AGE'] = 50
    pd.testing.assert_frame_equal(production, original)


def test_concept_seed_is_repeatable_and_id_selection_independent_of_row_order(production):
    first = build_concept_scenario(production, seed=42)
    second = build_concept_scenario(production, seed=42)
    pd.testing.assert_frame_equal(first, second)
    shuffled = production.sample(frac=1, random_state=9)
    actual = build_concept_scenario(shuffled, seed=42)
    pd.testing.assert_frame_equal(actual.sort_values('ID'), first.sort_values('ID'))
    assert not first[TARGET].equals(build_concept_scenario(production, seed=17)[TARGET])


@pytest.mark.parametrize('case', ['no_labels', 'no_delay_group', 'no_eligible_labels', 'too_small'])
def test_concept_requires_labels_and_effective_intervention(production, case):
    if case == 'no_labels':
        production = production.drop(columns=TARGET)
    elif case == 'no_delay_group':
        production['PAY_0'] = 0
    elif case == 'no_eligible_labels':
        production[TARGET] = 0
    else:
        production = production.iloc[[0, 1, 16, 17]]
    with pytest.raises(ValueError):
        build_concept_scenario(production)


@pytest.mark.parametrize('seed', [-1, True, 1.2, '42', None])
def test_concept_rejects_invalid_seed(production, seed):
    with pytest.raises(ValueError):
        build_concept_scenario(production, seed=seed)


def test_concept_validates_input_contract(production):
    production.loc[0, 'AGE'] = 18
    with pytest.raises(InvalidBatch):
        build_concept_scenario(production)
