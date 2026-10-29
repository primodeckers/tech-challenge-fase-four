import numpy as np
import pytest
import pandas as pd
import json
import hashlib
from pathlib import Path

from credit_monitor.drift import calibrate_policy, compare_drift, holm_adjust, ks_comparison, population_stability_index
from credit_monitor.contracts import InvalidBatch
from credit_monitor.baseline import FEATURES
from credit_monitor.drift_report import export_evidently
from credit_monitor.scenarios import build_scenarios
from credit_monitor.drift_run import run_drift


def test_psi_is_zero_for_identical_samples():
    values = np.arange(400)
    result = population_stability_index(values, values)
    assert result["status"] == "ok"
    assert result["psi"] == 0
    assert sum(result["reference_counts"]) == 400


def test_psi_matches_hand_calculation_with_fixed_bins_and_pseudocounts():
    result = population_stability_index([0, 0, 1, 1], [0, 1, 1, 1],
                                       cutpoints=[0.5], min_samples=1)
    # Smoothing0.5: p=(0.5,0.5), q=(0.3,0.7).
    assert result["psi"] == pytest.approx(0.2 * np.log(7 / 3))
    assert result["reference_counts"] == [2, 2]
    assert result["current_counts"] == [1, 3]


def test_psi_covers_extremes_and_zero_frequency_bins():
    result = population_stability_index([0, 0, 1, 1], [-10000, 10000, 10000, 10000],
                                       cutpoints=[-1, 0.5, 2], min_samples=1)
    assert sum(result["current_counts"]) == 4
    assert result["current_counts"] == [1, 0, 0, 3]
    assert np.isfinite(result["psi"]) and result["psi"] > 0


def test_quantile_cuts_separate_zero_mass_from_positive_values():
    reference = [0] * 390 + [1] * 10
    current = [0] * 200 + [1] * 200
    result = population_stability_index(reference, current)
    assert result["psi"] > 1
    assert len(result["reference_counts"]) >= 2


@pytest.mark.parametrize("bad", [[1, float('nan')], [1, float('inf')], [[1], [2]], []])
def test_psi_rejects_invalid_vectors_instead_of_dropping_values(bad):
    with pytest.raises(ValueError):
        population_stability_index(bad, [1, 2], min_samples=1)


def test_psi_reports_small_samples_and_constants_explicitly():
    small = population_stability_index([0, 1], [0, 2])
    assert small["status"] == "insufficient_samples" and small["psi"] is None
    constant = population_stability_index([1] * 200, [1] * 200)
    assert constant["status"] == "constant_reference" and constant["psi"] is None
    changed = population_stability_index([1] * 200, [2] * 200)
    assert changed["status"] == "constant_reference_changed" and changed["psi"] is None


@pytest.mark.parametrize("kwargs", [{"smoothing": 0}, {"smoothing": float('nan')},
    {"min_samples": 0}, {"cutpoints": [1, 1]}, {"cutpoints": [2, 1]}, {"cutpoints": [float('inf')]}])
def test_psi_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        population_stability_index(np.arange(400), np.arange(400), **kwargs)


def test_ks_identical_and_disjoint_samples_report_effect_and_pvalue():
    same = ks_comparison(np.arange(400), np.arange(400))
    assert same["statistic"] == 0 and same["pvalue"] == 1
    shifted = ks_comparison(np.arange(400), np.arange(400) + 1000)
    assert shifted["statistic"] == 1 and shifted["pvalue"] < 0.001
    assert shifted["n_reference"] == shifted["n_current"] == 400


def test_ks_reports_ties_and_small_samples_without_hiding_them():
    tied = ks_comparison([0] * 300, [1] * 300)
    assert tied["ties"] is True
    assert tied["pvalue_method"] == "asymptotic_continuous_approximation"
    small = ks_comparison([0, 1], [0, 2])
    assert small["status"] == "insufficient_samples"
    assert small["statistic"] is None and small["pvalue"] is None
    with pytest.raises(ValueError):
        ks_comparison([float('nan')], [1])


def test_holm_adjustment_matches_hand_calculation_and_preserves_names():
    assert holm_adjust({"c": 0.03, "a": 0.01, "b": 0.02}) == {
        "c": pytest.approx(0.04), "a": pytest.approx(0.03), "b": pytest.approx(0.04)}
    assert holm_adjust({"a": 0.8, "b": 0.9}) == {"a": 1.0, "b": 1.0}
    assert holm_adjust({}) == {}


@pytest.mark.parametrize("p", [-0.1, 1.1, float('nan')])
def test_holm_rejects_invalid_pvalues(p):
    with pytest.raises(ValueError):
        holm_adjust({"feature": p})


@pytest.fixture
def reference():
    n = 400
    values = {"ID": np.arange(n), "LIMIT_BAL": np.arange(1, n+1)*1000, "SEX": np.ones(n),
              "EDUCATION": np.ones(n), "MARRIAGE": np.ones(n), "AGE": np.full(n, 40)}
    for col in ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"):
        values[col] = np.zeros(n)
    for i in range(1, 7):
        values[f"BILL_AMT{i}"] = np.arange(n)*10
        values[f"PAY_AMT{i}"] = np.arange(1,n+1)*100
    return pd.DataFrame(values, dtype="int64")


@pytest.fixture
def calibration(reference):
    result = reference.copy()
    result["ID"] += 10000
    return result


def test_policy_is_json_safe_repeatable_and_uses_reference_bins(reference, calibration):
    first = calibrate_policy(reference, calibration, n_resamples=20)
    second = calibrate_policy(reference, calibration, n_resamples=20)
    assert json.loads(json.dumps(first, allow_nan=False)) == first == second
    assert first["features"]["LIMIT_BAL"]["cutpoints"] == population_stability_index(
        reference.LIMIT_BAL, calibration.LIMIT_BAL)["cutpoints"]
    assert first["calibration_rows"] == 400
    assert first["n_resamples"] == 20
    assert first["features"]["LIMIT_BAL"]["psi_threshold"] > 0


def test_comparison_reports_identical_control_and_two_known_changes(reference, calibration):
    policy = calibrate_policy(reference, calibration, n_resamples=20)
    control = compare_drift(reference, calibration, policy)
    assert control["alerted_features"] == []
    changed = calibration.copy()
    changed["LIMIT_BAL"] *= 3
    changed["PAY_AMT1"] //= 2
    result = compare_drift(reference, changed, policy)
    assert set(result["alerted_features"]) == {"LIMIT_BAL", "PAY_AMT1"}
    assert result["drift_detected"] is True
    assert result["features"]["LIMIT_BAL"]["ks_pvalue_holm"] <= 0.05


def test_constant_reference_change_is_not_reported_as_no_drift(reference, calibration):
    policy = calibrate_policy(reference, calibration, n_resamples=20)
    calibration["AGE"] = 41
    result = compare_drift(reference, calibration, policy)
    feature = result["features"]["AGE"]
    assert feature["status"] == "constant_reference_changed"
    assert feature["alert"] is True and feature["psi"] is None


def test_comparison_rejects_modified_reference_and_invalid_lot(reference, calibration):
    policy = calibrate_policy(reference, calibration, n_resamples=20)
    changed_reference = reference.copy()
    changed_reference.loc[0, "LIMIT_BAL"] = 999
    with pytest.raises(ValueError, match="referencia"):
        compare_drift(changed_reference, calibration, policy)
    calibration["LIMIT_BAL"] = calibration.LIMIT_BAL.astype("Int64")
    calibration.loc[0, "LIMIT_BAL"] = pd.NA
    with pytest.raises(InvalidBatch):
        compare_drift(reference, calibration, policy)


def test_small_or_different_size_lot_is_unavailable_not_false_success(reference, calibration):
    policy = calibrate_policy(reference, calibration, n_resamples=20)
    small = compare_drift(reference, calibration.iloc[:10], policy)
    assert small["status"] == "insufficient_samples" and small["drift_detected"] is None
    other_size = compare_drift(reference, calibration.iloc[:300], policy)
    assert other_size["status"] == "uncalibrated_sample_size"
    assert other_size["drift_detected"] is None


@pytest.mark.parametrize("kwargs", [{"n_resamples": 0}, {"quantile": 1}, {"alpha": 0}])
def test_policy_rejects_invalid_configuration(reference, calibration, kwargs):
    with pytest.raises(ValueError):
        calibrate_policy(reference, calibration, **kwargs)


def test_policy_rejects_overlapping_ids_and_too_small_calibration(reference, calibration):
    with pytest.raises(ValueError, match="IDs"):
        calibrate_policy(reference, reference, n_resamples=20)
    with pytest.raises(ValueError):
        calibrate_policy(reference, calibration.iloc[:10], n_resamples=20)


def test_evidently_html_and_metrics_use_same_policy_and_only_model_features(reference, calibration, tmp_path, monkeypatch):
    from evidently import Dataset
    original = Dataset.from_pandas
    observed = []

    def inspect_dataset(data, *args, **kwargs):
        observed.append(set(data.columns))
        return original(data, *args, **kwargs)

    monkeypatch.setattr(Dataset, "from_pandas", inspect_dataset)
    policy = calibrate_policy(reference, calibration, n_resamples=20)
    calibration["LIMIT_BAL"] *= 3
    calibration["PAY_AMT1"] //= 2
    output = tmp_path / "drift.html"
    result = export_evidently(reference, calibration, policy, output)
    assert output.stat().st_size > 1000
    assert "LIMIT_BAL" in output.read_text(encoding="utf-8")
    assert set(result["comparison"]["alerted_features"]) == {"LIMIT_BAL", "PAY_AMT1"}
    metrics = result["evidently"]["metrics"]
    count = next(metric for metric in metrics if metric["config"]["type"].endswith("DriftedColumnsCount"))
    assert count["value"]["count"] == 2
    assert len(observed) == 2 and all(columns == set(FEATURES) for columns in observed)


def test_evidently_reports_control_and_refuses_unavailable_or_existing_output(reference, calibration, tmp_path):
    policy = calibrate_policy(reference, calibration, n_resamples=20)
    output = tmp_path / "control.html"
    result = export_evidently(reference, calibration, policy, output)
    assert result["comparison"]["alerted_features"] == []
    saved = output.read_bytes()
    with pytest.raises(FileExistsError):
        export_evidently(reference, calibration, policy, output)
    assert output.read_bytes() == saved
    with pytest.raises(ValueError, match="indisponivel"):
        export_evidently(reference, calibration.iloc[:10], policy, tmp_path / "small.html")
    assert not (tmp_path / "small.html").exists()


@pytest.fixture
def inputs(reference, calibration, tmp_path):
    data, scenarios = tmp_path / "data", tmp_path / "scenarios"
    data.mkdir(); scenarios.mkdir()
    production = calibration.copy()
    production["ID"] += 10000
    manifest = {"partitions": {}}
    for name, frame in [("reference", reference), ("calibration", calibration), ("production", production)]:
        path = data / f"{name}.csv"
        frame.to_csv(path, index=False)
        manifest["partitions"][name] = {"file": path.name, "rows": len(frame),
                                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    payload = json.dumps(manifest).encode()
    (data / "manifest.json").write_bytes(payload)
    origin = {"stage": "S04", "data_manifest_sha256": hashlib.sha256(payload).hexdigest(), "files": {}}
    for name, frame in build_scenarios(production).items():
        frame.to_csv(scenarios / f"{name}.csv", index=False)
        pd.DataFrame({"ID": frame.ID, "probability_default": 0.2, "prediction": 0}).to_csv(
            scenarios / f"{name}_predictions.csv", index=False)
    for path in scenarios.glob('*.csv'):
        origin["files"][path.name] = {"rows": len(production), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (scenarios / "manifest.json").write_text(json.dumps(origin))
    return data, scenarios


def test_drift_execution_reproduces_numeric_outputs_and_preserves_inputs(inputs, tmp_path):
    data, scenarios = inputs
    before = {path: path.read_bytes() for folder in inputs for path in folder.iterdir()}
    a, b = tmp_path / "run1", tmp_path / "run2"
    run_drift(data, scenarios, a, n_resamples=20)
    run_drift(data, scenarios, b, n_resamples=20)
    for name in ["policy.json", "comparisons.json"]:
        assert (a / name).read_bytes() == (b / name).read_bytes()
    metrics = json.loads((a / "comparisons.json").read_text())
    assert metrics["control"]["alerted_features"] == []
    assert set(metrics["data_drift"]["alerted_features"]) == {"LIMIT_BAL", "PAY_AMT1"}
    manifest = json.loads((a / "manifest.json").read_text())
    for name, info in manifest["files"].items():
        assert hashlib.sha256((a / name).read_bytes()).hexdigest() == info["sha256"]
    assert (a / "control.html").stat().st_size > 1000
    assert (a / "data_drift.html").stat().st_size > 1000
    assert (a / "overview.html").exists()
    for path, content in before.items():
        assert path.read_bytes() == content


@pytest.mark.parametrize("part", ["calibration", "control", "lineage"])
def test_drift_execution_rejects_tampered_inputs_before_success(inputs, tmp_path, part):
    data, scenarios = inputs
    path = data / "calibration.csv" if part == "calibration" else scenarios / ("control.csv" if part == "control" else "manifest.json")
    if part == "lineage":
        origin = json.loads(path.read_text())
        origin["data_manifest_sha256"] = "foreign data manifest"
        path.write_text(json.dumps(origin))
    else:
        with path.open("ab") as stream:
            stream.write(b" ")
    with pytest.raises(ValueError):
        run_drift(data, scenarios, tmp_path / "run", n_resamples=20)
    assert not (tmp_path / "run/manifest.json").exists()


def test_drift_execution_does_not_overwrite_existing_run(inputs, tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "manifest.json").write_text("keep")
    with pytest.raises(FileExistsError):
        run_drift(*inputs, output, n_resamples=20)
    assert (output / "manifest.json").read_text() == "keep"


def test_report_failure_does_not_leave_success_manifest(inputs, tmp_path, monkeypatch):
    from credit_monitor import drift_run

    def fail(*args, **kwargs):
        raise OSError("synthetic report failure")

    monkeypatch.setattr(drift_run, "export_evidently", fail, raising=False)
    with pytest.raises(OSError):
        run_drift(*inputs, tmp_path / "run", n_resamples=20)
    assert not (tmp_path / "run/manifest.json").exists()
