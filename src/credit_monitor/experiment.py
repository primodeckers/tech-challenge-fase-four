"""Execucao reproduzivel da S03; nao avalia nem ajusta usando producao."""

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd

from credit_monitor.baseline import (FEATURES, calibration_without_seen_profiles,
                                     evaluate_binary, predict_scores, select_threshold, train_models)
from credit_monitor.data import TARGET
from credit_monitor.model_store import load_models, save_models


def _read_partition(data_dir, name, manifest):
    info = manifest["partitions"][name]
    if info["file"] != f"{name}.csv":
        raise ValueError("Nome de particao inesperado no manifesto")
    payload = (data_dir / f"{name}.csv").read_bytes()
    if hashlib.sha256(payload).hexdigest() != info["sha256"]:
        raise ValueError(f"SHA-256 divergente na particao {name}")
    frame = pd.read_csv(BytesIO(payload))
    if len(frame) != info["rows"]:
        raise ValueError(f"Contagem divergente na particao {name}")
    return frame

def run_baseline(data_dir, output_dir):
    """Ajusta na referencia e escolhe limiares na calibracao; producao fica reservada."""
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Escolha uma pasta nova para preservar os artefatos existentes")
    manifest_bytes = (data_dir / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    reference = _read_partition(data_dir, "reference", manifest)
    calibration = _read_partition(data_dir, "calibration", manifest)
    clean, audit = calibration_without_seen_profiles(reference, calibration)
    # Validar a elegibilidade da calibracao antes de gastar tempo treinando.
    if set(clean[TARGET].unique()) != {0, 1}:
        raise ValueError("Calibracao retida precisa das duas classes")
    models = train_models(reference)
    report = {
        "stage": "S03", "reference_rows": len(reference), "overlap_audit": audit,
        "features": list(FEATURES), "production_used": False,
        "evaluation_population": "calibration_without_training_feature_profiles",
        "threshold_policy": "max_f1_grid_0.05_to_0.95_step_0.01_tie_highest",
        "limitations": ["Threshold selection and evaluation share calibration; metrics are optimistic development estimates.",
                        "Exact-feature exclusions do not establish independent people or temporal generalization.",
                        "Financial costs and probability calibration were not optimized."],
        "data_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "input_sha256": {name: manifest["partitions"][name]["sha256"] for name in ("reference", "calibration")},
        "code_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                        for name in ("baseline.py", "contracts.py", "model_store.py", "experiment.py")},
        "model_configuration": {
            "algorithm": "StandardScaler + LogisticRegression", "solver": "lbfgs", "C": 1.0,
            "max_iter": 2000, "random_state": 42, "class_weight": None,
            "iterations": models.model.named_steps["classifier"].n_iter_.tolist(),
            "dummy_strategy": "prior",
        },
    }
    thresholds = {}
    clean_x, full_x = clean.drop(columns=TARGET), calibration.drop(columns=TARGET)
    original_scores = {}
    for name in ("model", "dummy"):
        estimator = getattr(models, name)
        scores = predict_scores(estimator, clean_x)
        original_scores[name] = scores
        threshold = select_threshold(clean[TARGET], scores)
        thresholds[name] = threshold
        report[name] = {
            "selected": evaluate_binary(clean[TARGET], scores, threshold),
            "at_0_5": evaluate_binary(clean[TARGET], scores, 0.5),
            "full_calibration_sensitivity": evaluate_binary(
                calibration[TARGET], predict_scores(estimator, full_x), threshold),
        }
    save_models(models, output_dir, thresholds=thresholds, metadata=report)
    try:
        restored, saved_manifest = load_models(output_dir, trusted=True)
        for name in ("model", "dummy"):
            loaded_scores = predict_scores(getattr(restored, name), clean_x)
            if not np.array_equal(loaded_scores, original_scores[name]):
                raise ValueError("Probabilidades divergentes apos persistencia")
        if saved_manifest["thresholds"] != thresholds:
            raise ValueError("Limiares divergentes apos persistencia")
        (output_dir / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    except Exception:
        (output_dir / "manifest.json").unlink(missing_ok=True)
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/baseline"))
    args = parser.parse_args()
    report = run_baseline(args.data_dir, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "overlap_audit": report["overlap_audit"],
                      "model": report["model"]["selected"], "dummy": report["dummy"]["selected"]}, indent=2))


if __name__ == "__main__":
    main()
