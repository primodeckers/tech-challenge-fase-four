"""Exportacao reproduzivel dos cenarios S04 e predicoes congeladas."""

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd

from credit_monitor.baseline import FEATURES
from credit_monitor.model_store import load_models
from credit_monitor.scenarios import TRANSFORMATIONS, build_scenarios, copy_control, predict_scenarios


def _verified_partition(data_dir, name, manifest):
    info = manifest["partitions"][name]
    if info["file"] != f"{name}.csv":
        raise ValueError("Nome de arquivo inesperado no manifesto")
    payload = (data_dir / info["file"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != info["sha256"]:
        raise ValueError(f"SHA-256 divergente na particao {name}")
    frame = copy_control(pd.read_csv(BytesIO(payload)))
    if len(frame) != info["rows"]:
        raise ValueError("Contagem divergente do manifesto")
    return frame

def run_scenarios(data_dir, model_dir, output_dir, *, trusted_model=False):
    """Gera somente controle, intervencao em X e predicoes do artefato S03.

    Manifesto e ultimo arquivo escrito; uma falha anterior nao publica sucesso.
    Dados individuais ficam nos CSVs locais, nao nos resumos do terminal.
    """
    data_dir, model_dir, output_dir = Path(data_dir), Path(model_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Escolha uma pasta nova para preservar execucoes existentes")
    models, model_manifest = load_models(model_dir, trusted=trusted_model)
    manifest_bytes = (data_dir / "manifest.json").read_bytes()
    data_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if model_manifest["metadata"].get("data_manifest_sha256") != data_sha:
        raise ValueError("Dados nao correspondem a origem registrada no artefato S03")
    source = json.loads(manifest_bytes)
    production = _verified_partition(data_dir, "production", source)
    reference = _verified_partition(data_dir, "reference", source)
    if set(reference.ID) & set(production.ID):
        raise ValueError("IDs de referencia e producao precisam ser disjuntos")
    scenarios = build_scenarios(production)
    threshold = model_manifest["thresholds"]["model"]
    predictions = predict_scenarios(models.model, threshold, scenarios)
    known = pd.MultiIndex.from_frame(reference.loc[:, FEATURES])
    shared = pd.MultiIndex.from_frame(production.loc[:, FEATURES]).isin(known)
    control, drift = scenarios["control"], scenarios["data_drift"]
    first, second = predictions["control"], predictions["data_drift"]
    summary = {}
    for name, prediction in predictions.items():
        summary[name] = {
            "rows": len(prediction),
            "positive_count": int(prediction.prediction.sum()),
            "positive_rate": float(prediction.prediction.mean()),
            "mean_probability": float(prediction.probability_default.mean()),
        }
    manifest = {
        "stage": "S04", "format_version": 1, "seed": None,
        "randomness": "none; integer rational scaling",
        "transformations": {column: {"numerator": ratio[0], "denominator": ratio[1], "rounding": "floor"}
                            for column, ratio in TRANSFORMATIONS.items()},
        "threshold": threshold, "model_sha256": model_manifest["model_sha256"],
        "model_manifest_sha256": hashlib.sha256((model_dir / "manifest.json").read_bytes()).hexdigest(),
        "data_manifest_sha256": data_sha,
        "input_sha256": {name: source["partitions"][name]["sha256"] for name in ("production", "reference")},
        "versions": model_manifest["versions"],
        "code_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                        for name in ("scenarios.py", "scenario_run.py", "baseline.py", "contracts.py", "model_store.py")},
        "changed_cells": {column: int((control[column] != drift[column]).sum()) for column in TRANSFORMATIONS},
        "production_profiles_seen_in_reference": int(shared.sum()),
        "scenario_summary": summary,
        "changed_predictions": int((first.prediction != second.prediction).sum()),
        "mean_absolute_probability_change": float(np.abs(first.probability_default - second.probability_default).mean()),
        "limitations": ["Synthetic intervention on X preserves historical Y; not proof of economic causality or pure covariate shift.",
                        "Prediction changes do not measure accuracy; no performance evaluation or drift detector in S04.",
                        "Production retains exact feature profiles seen in training; not an independent-person test."],
        "files": {},
    }
    payloads = {f"{name}.csv": frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
                for name, frame in scenarios.items()}
    payloads.update({f"{name}_predictions.csv": frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
                     for name, frame in predictions.items()})
    for name, payload in payloads.items():
        manifest["files"][name] = {"sha256": hashlib.sha256(payload).hexdigest(), "rows": len(production)}
    serialized = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, payload in payloads.items():
        (output_dir / name).write_bytes(payload)
    (output_dir / "manifest.json").write_text(serialized, encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/baseline-s03-20261026"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/scenarios-s04"))
    parser.add_argument("--trusted-model", action="store_true", help="Somente artefato local de origem conhecida")
    args = parser.parse_args()
    result = run_scenarios(args.data_dir, args.model_dir, args.output_dir, trusted_model=args.trusted_model)
    print(json.dumps({"output_dir": str(args.output_dir), "threshold": result["threshold"],
                      "changed_cells": result["changed_cells"], "scenario_summary": result["scenario_summary"],
                      "changed_predictions": result["changed_predictions"]}, indent=2))


if __name__ == "__main__":
    main()
