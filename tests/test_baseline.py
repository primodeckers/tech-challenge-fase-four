import numpy as np
import pandas as pd
import pytest

from credit_monitor.baseline import FEATURES, calibration_without_seen_profiles, predict_scores, train_models
from credit_monitor.contracts import InvalidBatch
from credit_monitor.data import TARGET


@pytest.fixture
def reference():
    rng = np.random.default_rng(12)
    n = 80
    values = {"ID": np.arange(n), "LIMIT_BAL": rng.integers(1, 20, n) * 10000,
              "SEX": rng.integers(1, 3, n), "EDUCATION": rng.integers(0, 7, n),
              "MARRIAGE": rng.integers(0, 4, n), "AGE": rng.integers(19, 80, n)}
    for col in ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"):
        values[col] = rng.integers(-2, 9, n)
    for i in range(1, 7):
        values[f"BILL_AMT{i}"] = rng.integers(-1000, 10000, n)
        values[f"PAY_AMT{i}"] = rng.integers(0, 5000, n)
    values[TARGET] = np.tile([0, 0, 0, 1], n // 4)
    return pd.DataFrame(values, dtype="int64")


@pytest.fixture
def calibration(reference):
    frame = reference.iloc[:20].copy().reset_index(drop=True)
    frame["ID"] += 1000
    frame["LIMIT_BAL"] += 1000
    return frame


def test_calibration_excludes_seen_features_regardless_of_label_or_audit(reference, calibration):
    calibration.loc[0, list(FEATURES)] = reference.loc[0, list(FEATURES)]
    calibration.loc[0, TARGET] = 1
    calibration.loc[0, "SEX"] = 3 - reference.loc[0, "SEX"]
    before = calibration.copy(deep=True)
    clean, audit = calibration_without_seen_profiles(reference, calibration)
    assert len(clean) == 19
    assert 1000 not in set(clean.ID)
    assert audit == {"original_rows": 20, "excluded_seen_profiles": 1, "retained_rows": 19}
    pd.testing.assert_frame_equal(calibration, before)


def test_calibration_rejects_overlapping_ids(reference, calibration):
    calibration.loc[0, "ID"] = 0
    with pytest.raises(ValueError, match="ID"):
        calibration_without_seen_profiles(reference, calibration)


def test_calibration_rejects_when_no_unseen_rows_remain(reference):
    other = reference.copy()
    other["ID"] += 1000
    with pytest.raises(ValueError, match="calibracao"):
        calibration_without_seen_profiles(reference, other)


def test_training_fits_scaler_only_on_reference_and_uses_twenty_features(reference):
    before = reference.copy(deep=True)
    models = train_models(reference)
    assert len(models.model.feature_names_in_) == 20
    assert not {"ID", "SEX", "EDUCATION", "MARRIAGE", TARGET} & set(models.model.feature_names_in_)
    np.testing.assert_allclose(models.model.named_steps["scale"].mean_, reference[list(FEATURES)].mean())
    np.testing.assert_allclose(models.dummy.class_prior_, [0.75, 0.25])
    pd.testing.assert_frame_equal(reference, before)


def test_predictions_are_reproducible_and_ignore_id_and_audit_columns(reference, calibration):
    first = train_models(reference)
    second = train_models(reference)
    frame = calibration.drop(columns=TARGET)
    p = predict_scores(first.model, frame)
    assert p.shape == (20,)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    np.testing.assert_allclose(p, predict_scores(second.model, frame), atol=1e-12)
    changed = frame.copy()
    changed["ID"] += 10000
    changed["SEX"] = 3 - changed.SEX
    changed["EDUCATION"] = 0
    changed["MARRIAGE"] = 0
    np.testing.assert_array_equal(p, predict_scores(first.model, changed[changed.columns[::-1]]))
    np.testing.assert_allclose(predict_scores(first.dummy, frame), np.full(20, 0.25))


def test_training_rejects_single_class_and_invalid_data(reference):
    reference[TARGET] = 0
    with pytest.raises(ValueError, match="classes"):
        train_models(reference)
    reference.loc[0, "LIMIT_BAL"] = 0
    with pytest.raises(InvalidBatch):
        train_models(reference)


def test_invalid_inference_never_calls_predictor(calibration):
    class NeverCalled:
        def predict_proba(self, frame):
            pytest.fail("inference ran on invalid batch")

    with pytest.raises(InvalidBatch):
        predict_scores(NeverCalled(), calibration)  # alvo presente


def test_prediction_does_not_fit_scaler_on_calibration(reference, calibration):
    models = train_models(reference)
    before = models.model.named_steps["scale"].mean_.copy()
    calibration["LIMIT_BAL"] *= 100
    predict_scores(models.model, calibration.drop(columns=TARGET))
    np.testing.assert_array_equal(models.model.named_steps["scale"].mean_, before)
