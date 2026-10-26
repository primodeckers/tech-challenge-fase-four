import numpy as np
import pandas as pd
import pytest
import json

from credit_monitor.baseline import FEATURES, calibration_without_seen_profiles, evaluate_binary, predict_scores, select_threshold, train_models
from credit_monitor.contracts import InvalidBatch
from credit_monitor.data import TARGET
from credit_monitor.model_store import load_models, save_models


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


def test_binary_metrics_match_hand_calculated_example():
    metrics = evaluate_binary([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], threshold=0.4)
    assert metrics["rows"] == 4
    assert metrics["roc_auc"] == pytest.approx(0.75)
    assert metrics["average_precision"] == pytest.approx(5 / 6)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["brier"] == pytest.approx(0.158125)
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]


def test_threshold_maximizes_grid_f1_and_breaks_ties_towards_larger_value():
    assert select_threshold([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == 0.35
    assert select_threshold([0, 1], [0.2, 0.8]) == 0.8
    with pytest.raises(ValueError, match="classes"):
        select_threshold([0, 0], [0.2, 0.8])


@pytest.mark.parametrize("labels,scores", [([], []), ([0,1],[0.5]), ([0,2],[0.1,0.9]),
    ([0,1],[float('nan'),0.9]), ([0,1],[-0.1,0.9]), ([0,1],[0.1,1.1]),
    ([[0],[1]],[0.1,0.9]), ([0,1],[[0.1],[0.9]])])
def test_metrics_and_threshold_reject_invalid_inputs(labels, scores):
    with pytest.raises(ValueError):
        evaluate_binary(labels, scores, threshold=0.5)
    with pytest.raises(ValueError):
        select_threshold(labels, scores)


@pytest.mark.parametrize("threshold", [-1, 1.1, float('nan')])
def test_metrics_reject_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        evaluate_binary([0,1], [0.1,0.8], threshold)


def test_single_class_evaluation_marks_discrimination_metrics_unavailable():
    metrics = evaluate_binary([0,0], [0.1,0.2], 0.5)
    assert metrics["roc_auc"] is None
    assert metrics["average_precision"] is None
    assert metrics["confusion_matrix"] == [[2,0],[0,0]]


def test_saved_model_restores_identical_probabilities_and_decision(reference, calibration, tmp_path):
    models = train_models(reference)
    path = tmp_path / "model"
    thresholds = {"model": 0.35, "dummy": 0.25}
    save_models(models, path, thresholds=thresholds, metadata={"purpose": "test"})
    restored, manifest = load_models(path, trusted=True)
    assert manifest["thresholds"] == thresholds
    batch = calibration.drop(columns=TARGET)
    for name in ("model", "dummy"):
        p = predict_scores(getattr(models, name), batch)
        q = predict_scores(getattr(restored, name), batch)
        np.testing.assert_array_equal(p, q)
        np.testing.assert_array_equal(p >= thresholds[name], q >= manifest["thresholds"][name])


def test_loader_requires_explicit_trust_before_accessing_files(tmp_path):
    with pytest.raises(ValueError, match="trusted"):
        load_models(tmp_path)


@pytest.mark.parametrize("corruption", ["payload", "version", "features", "threshold"])
def test_loader_rejects_corruption_or_incompatible_metadata(reference, tmp_path, corruption):
    path = tmp_path / "model"
    save_models(train_models(reference), path, thresholds={"model": 0.35, "dummy": 0.25}, metadata={})
    if corruption == "payload":
        (path / "models.pkl").write_bytes(b"corrupt payload")
    else:
        manifest = json.loads((path / "manifest.json").read_text())
        if corruption == "version":
            manifest["versions"]["scikit-learn"] = "0.0.0"
        elif corruption == "features":
            manifest["features"] = ["ID"]
        else:
            manifest["thresholds"]["model"] = 2
        (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_models(path, trusted=True)


def test_save_never_overwrites_existing_artifacts(reference, tmp_path):
    path = tmp_path / "model"
    path.mkdir()
    (path / "keep.txt").write_text("existing")
    with pytest.raises(FileExistsError):
        save_models(train_models(reference), path, thresholds={"model": 0.35, "dummy": 0.25}, metadata={})
    assert (path / "keep.txt").read_text() == "existing"


def test_save_rejects_invalid_threshold_before_creating_directory(reference, tmp_path):
    path = tmp_path / "model"
    with pytest.raises(ValueError):
        save_models(train_models(reference), path, thresholds={"model": float('nan'), "dummy": 0.25}, metadata={})
    assert not path.exists()
