import numpy as np
import pandas as pd
import pytest

from credit_monitor.contracts import InvalidBatch, validate_batch
from credit_monitor.data import TARGET
from credit_monitor.scenarios import build_scenarios, copy_control


@pytest.fixture
def production():
    values = {"ID": [1001, 1002, 1003, 1004], "LIMIT_BAL": [10000, 20001, 30000, 40001],
              "SEX": [1, 2, 1, 2], "EDUCATION": [0, 5, 6, 1], "MARRIAGE": [0, 1, 3, 2],
              "AGE": [19, 30, 50, 100]}
    for col in ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"):
        values[col] = [-2, -1, 0, 8]
    for i in range(1, 7):
        values[f"BILL_AMT{i}"] = [-100, 0, 1000, 5000]
        values[f"PAY_AMT{i}"] = [0, 1, 1001, 2000]
    values[TARGET] = [0, 1, 0, 1]
    return pd.DataFrame(values, dtype="int64")


@pytest.mark.parametrize("labeled", [True, False])
def test_control_is_identical_and_independent_of_input(production, labeled):
    if not labeled:
        production = production.drop(columns=TARGET)
    before = production.copy(deep=True)
    control = copy_control(production)
    pd.testing.assert_frame_equal(control, before)
    control.loc[0, "LIMIT_BAL"] = 99999
    pd.testing.assert_frame_equal(production, before)


def test_control_rejects_invalid_production_instead_of_repairing(production):
    production.loc[0, "LIMIT_BAL"] = 0
    with pytest.raises(InvalidBatch):
        copy_control(production)
    assert production.loc[0, "LIMIT_BAL"] == 0


@pytest.mark.parametrize("labeled", [True, False])
def test_drift_changes_exactly_two_features_with_hand_calculated_values(production, labeled):
    if not labeled:
        production = production.drop(columns=TARGET)
    original = production.copy(deep=True)
    scenarios = build_scenarios(production)
    assert set(scenarios) == {"control", "data_drift"}
    control, drift = scenarios["control"], scenarios["data_drift"]
    pd.testing.assert_frame_equal(control, original)
    assert drift.LIMIT_BAL.tolist() == [15000, 30001, 45000, 60001]
    assert drift.PAY_AMT1.tolist() == [0, 0, 500, 1000]
    unchanged = [c for c in original if c not in ("LIMIT_BAL", "PAY_AMT1")]
    pd.testing.assert_frame_equal(drift[unchanged], original[unchanged])
    assert all(dtype == np.dtype("int64") for dtype in drift.dtypes)
    pd.testing.assert_frame_equal(production, original)
    validate_batch(drift, mode="training" if labeled else "inference")


def test_scenarios_are_repeatable_and_do_not_share_mutable_data(production):
    first, second = build_scenarios(production), build_scenarios(production)
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])
    first["data_drift"].loc[0, "AGE"] = 50
    assert first["control"].loc[0, "AGE"] == 19
    assert production.loc[0, "AGE"] == 19


def test_drift_rejects_overflow_without_silently_wrapping_or_mutating(production):
    production.loc[0, "LIMIT_BAL"] = np.iinfo(np.int64).max
    original = production.copy(deep=True)
    with pytest.raises(ValueError, match="int64"):
        build_scenarios(production)
    pd.testing.assert_frame_equal(production, original)


def test_drift_rejects_when_payment_feature_cannot_change(production):
    production["PAY_AMT1"] = 0
    with pytest.raises(ValueError, match="PAY_AMT1"):
        build_scenarios(production)
