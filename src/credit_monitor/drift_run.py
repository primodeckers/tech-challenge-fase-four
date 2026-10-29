"""Execucao S05: politica calibrada, comparacoes e relatorios locais."""

import argparse
import hashlib
from importlib.metadata import version
from io import BytesIO
import json
from pathlib import Path

import pandas as pd

from credit_monitor.data import TARGET
from credit_monitor.contracts import validate_batch
from credit_monitor.drift import calibrate_policy
from credit_monitor.drift_report import export_evidently, write_overview


def _read_verified(directory, filename, info):
    payload = (directory / filename).read_bytes()
    if hashlib.sha256(payload).hexdigest() != info["sha256"]:
        raise ValueError(f"SHA-256 divergente em {filename}")
    frame = pd.read_csv(BytesIO(payload))
    if len(frame) != info["rows"]:
        raise ValueError(f"Contagem divergente em {filename}")
    return frame


def _partition(directory, name, source):
    info = source["partitions"][name]
    if info["file"] != f"{name}.csv":
        raise ValueError("Nome de particao inesperado")
    frame = _read_verified(directory, info["file"], info)
    return validate_batch(frame, mode="training" if TARGET in frame else "inference")


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

def run_drift(data_dir, scenarios_dir, output_dir, *, n_resamples=200, seed=42):
    """Verifica fontes, salva politica ANTES de consumir cenarios e emite manifestofinal."""
    data_dir, scenarios_dir, output_dir = map(Path, (data_dir, scenarios_dir, output_dir))
    if output_dir.exists():
        raise FileExistsError("Escolha uma pasta nova para preservar resultados")
    source_bytes = (data_dir / "manifest.json").read_bytes()
    source = json.loads(source_bytes)
    origin_bytes = (scenarios_dir / "manifest.json").read_bytes()
    origin = json.loads(origin_bytes)
    if origin["stage"] != "S04" or origin["data_manifest_sha256"] != hashlib.sha256(source_bytes).hexdigest():
        raise ValueError("Origem dos cenarios nao corresponde aos dados")
    expected = {"control.csv", "data_drift.csv", "control_predictions.csv", "data_drift_predictions.csv"}
    if set(origin["files"]) != expected:
        raise ValueError("Manifesto de cenarios incompleto ou com caminhos inesperados")
    reference = _partition(data_dir, "reference", source)
    calibration = _partition(data_dir, "calibration", source)
    policy = calibrate_policy(reference, calibration, n_resamples=n_resamples, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "policy.json", policy)
    policy = json.loads((output_dir / "policy.json").read_text(encoding="utf-8"))
    # Apenas apos fixar politica, ler/validar os dados de producao e cenarios.
    production = _partition(data_dir, "production", source)
    if set(production.ID) & (set(reference.ID) | set(calibration.ID)):
        raise ValueError("IDs de producao sobrepostos a referencia/calibracao")
    files = {name: _read_verified(scenarios_dir, name, origin["files"][name]) for name in sorted(expected)}
    control, drift = files["control.csv"], files["data_drift.csv"]
    if not control.equals(production):
        raise ValueError("Controle nao e identico a producao original")
    unchanged = [column for column in control if column not in ("LIMIT_BAL", "PAY_AMT1")]
    if set(drift.columns) != set(control.columns) or not control[unchanged].equals(drift[unchanged]):
        raise ValueError("Cenarios perderam pareamento ou alteraram outras colunas")
    comparisons, snapshots = {}, {}
    for name, frame in [("control", control), ("data_drift", drift)]:
        result = export_evidently(reference, frame, policy, output_dir / f"{name}.html")
        comparisons[name], snapshots[name] = result["comparison"], result["evidently"]
    _write_json(output_dir / "comparisons.json", comparisons)
    _write_json(output_dir / "evidently.json", snapshots)
    write_overview(comparisons, output_dir / "overview.html")
    manifest = {"stage": "S05", "format_version": 1,
        "data_manifest_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "scenario_manifest_sha256": hashlib.sha256(origin_bytes).hexdigest(),
        "versions": {name: version(name) for name in ("numpy", "pandas", "scipy", "evidently")},
        "code_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                        for name in ("drift.py", "drift_report.py", "drift_run.py", "contracts.py")},
        "alerts": {name: value["alerted_features"] for name, value in comparisons.items()},
        "files": {path.name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                  for path in sorted(output_dir.iterdir()) if path.is_file()},
        "reproducibility": "policy and statistical results deterministic; HTML may contain Evidently runtime IDs/timestamps"}
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--scenarios-dir", type=Path, default=Path("artifacts/scenarios-s04-20261026"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/drift-s05"))
    args = parser.parse_args()
    manifest = run_drift(args.data_dir, args.scenarios_dir, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "alerts": manifest["alerts"]}, indent=2))


if __name__ == "__main__":
    main()
