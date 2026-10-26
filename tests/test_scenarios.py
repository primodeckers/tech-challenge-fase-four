import numpy as np
import pandas as pd
import pytest

from credit_monitor.contracts import InvalidBatch, validate_batch
from credit_monitor.baseline import predict_scores, train_models
from credit_monitor.data import TARGET
from credit_monitor.scenarios import build_scenarios, copy_control, predict_scenarios


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


def test_predictions_use_frozen_model_threshold_and_preserve_pairing(production):
    model = train_models(production).model
    coef = model.named_steps["classifier"].coef_.copy()
    mean = model.named_steps["scale"].mean_.copy()
    scenarios = build_scenarios(production)
    first = predict_scenarios(model, 0.27, scenarios)
    repeated = predict_scenarios(model, 0.27, scenarios)
    for name in scenarios:
        result = first[name]
        pd.testing.assert_frame_equal(result, repeated[name])
        assert result.ID.tolist() == production.ID.tolist()
        assert result.columns.tolist() == ["ID", "probability_default", "prediction"]
        p = result.probability_default.to_numpy()
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        np.testing.assert_array_equal(result.prediction, (p >= 0.27).astype(int))
        np.testing.assert_array_equal(p, predict_scores(model, scenarios[name].drop(columns=TARGET)))
    np.testing.assert_array_equal(coef, model.named_steps["classifier"].coef_)
    np.testing.assert_array_equal(mean, model.named_steps["scale"].mean_)


def test_unlabeled_scenarios_produce_same_predictions(production):
    model = train_models(production).model
    labeled = predict_scenarios(model, 0.27, build_scenarios(production))
    unlabeled = predict_scenarios(model, 0.27, build_scenarios(production.drop(columns=TARGET)))
    for name in labeled:
        pd.testing.assert_frame_equal(labeled[name], unlabeled[name])


@pytest.mark.parametrize("case", ["invalid_drift", "mismatched_ids", "mismatched_labels", "missing_scenario"])
def test_invalid_pair_is_rejected_before_any_inference(production, case):
    scenarios = build_scenarios(production)
    if case == "invalid_drift":
        scenarios["data_drift"].loc[0, "AGE"] = 18
    elif case == "mismatched_ids":
        scenarios["data_drift"].loc[0, "ID"] = 99999
    elif case == "mismatched_labels":
        scenarios["data_drift"].loc[0, TARGET] = 1
    else:
        del scenarios["data_drift"]

    class NeverCalled:
        classes_ = [0, 1]

        def predict_proba(self, frame):
            pytest.fail("Prediction reached invalid scenario pair")

    with pytest.raises((ValueError, InvalidBatch)):
        predict_scenarios(NeverCalled(), 0.27, scenarios)


@pytest.mark.parametrize("threshold", [-1, 2, float('nan')])
def test_invalid_threshold_is_rejected_before_inference(production, threshold):
    with pytest.raises(ValueError):
        predict_scenarios(None, threshold, build_scenarios(production))
