"""Aquisicao e particionamento reproduzivel da base UCI 350."""

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd
from sklearn.model_selection import train_test_split

SOURCE_URL = "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients"
DOWNLOAD_URL = "https://archive.ics.uci.edu/static/public/350/default%2Bof%2Bcredit%2Bcard%2Bclients.zip"
# Bytes obtidos da fonte oficial em 13/10/2026. Mudancas exigem revisao explicita.
ARCHIVE_SHA256 = "56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602"
XLS_NAME = "default of credit card clients.xls"
TARGET = "default_next_month"


def normalize_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Mantem nomes UCI das features e explicita o nome do alvo."""
    if "default payment next month" not in frame.columns or TARGET in frame.columns:
        raise ValueError("Cabecalho do alvo UCI ausente ou ambiguo")
    return frame.rename(columns={"default payment next month": TARGET}).copy()


def profile_data(frame: pd.DataFrame) -> dict:
    """Resumo agregado de inspecao, sem inferir dominios nem corrigir valores."""
    categorical = ["SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    numeric = frame.select_dtypes(include="number")
    return {
        "rows": len(frame),
        "columns": frame.columns.tolist(),
        "dtypes": {c: str(t) for c, t in frame.dtypes.items()},
        "nulls": {c: int(v) for c, v in frame.isna().sum().items()},
        "duplicate_ids": int(frame.ID.duplicated().sum()),
        "duplicate_rows_without_id": int(frame.drop(columns="ID").duplicated().sum()),
        "numeric_ranges": {
            c: {"min": float(numeric[c].min()), "max": float(numeric[c].max())}
            for c in numeric if c != "ID"
        },
        "categorical_codes": {
            c: sorted(int(v) for v in frame[c].dropna().unique())
            for c in categorical if c in frame
        },
        "negative_counts": {c: int((numeric[c] < 0).sum()) for c in numeric if c != "ID"},
        "target_counts": {str(k): int(v) for k, v in frame[TARGET].value_counts().sort_index().items()},
    }


def split_data(frame: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Divisao estratificada 60/20/20 por ID; nao representa cronologia real."""
    if not {"ID", TARGET}.issubset(frame.columns):
        raise ValueError("Colunas ID e alvo sao obrigatorias")
    if len(frame) < 5000:
        raise ValueError("A base precisa de pelo menos 5000 registros")
    if frame.ID.isna().any() or not frame.ID.is_unique:
        raise ValueError("ID deve ser unico e nao nulo")
    if frame[TARGET].isna().any() or set(frame[TARGET].unique()) != {0, 1}:
        raise ValueError("O alvo deve conter as duas classes binarias 0 e 1")
    ordered = frame.sort_values("ID").reset_index(drop=True)
    reference, holdout = train_test_split(
        ordered, test_size=0.4, random_state=seed, stratify=ordered[TARGET]
    )
    calibration, production = train_test_split(
        holdout, test_size=0.5, random_state=seed, stratify=holdout[TARGET]
    )
    return {
        name: part.sort_values("ID").reset_index(drop=True)
        for name, part in zip(
            ("reference", "calibration", "production"),
            (reference, calibration, production), strict=True,
        )
    }


def acquire_data(raw_dir: Path) -> Path:
    """Baixa/cacheia o ZIP verificado; extrai apenas o XLS nominal, sem caminhos externos."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "uci-credit.zip"
    if archive.exists():
        payload = archive.read_bytes()
    else:
        with urlopen(DOWNLOAD_URL, timeout=60) as response:
            payload = response.read()
    if hashlib.sha256(payload).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("SHA-256 do ZIP diverge da fonte revisada; nenhum dado sera usado")
    if not archive.exists():
        archive.write_bytes(payload)
    destination = raw_dir / XLS_NAME
    with ZipFile(archive) as source:
        destination.write_bytes(source.read(XLS_NAME))
    return destination


def write_partitions(frame: pd.DataFrame, output_dir: Path, seed: int = 42, *, source: dict | None = None) -> dict:
    """Persiste CSVs e manifesto deterministico; sem ajustar transformacoes nos dados."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Um manifesto antigo nao deve indicar sucesso se uma reexecucao falhar.
    manifest_path = output_dir / "manifest.json"
    manifest_path.unlink(missing_ok=True)
    parts = split_data(frame, seed)
    manifest = {
        "seed": seed,
        "target": TARGET,
        "split_method": "stratified_random_by_unique_id_60_20_20",
        "temporal_split": False,
        "input_rows": len(frame),
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("pandas", "scikit-learn", "xlrd")},
        "partitions": {},
    }
    if source is not None:
        manifest["source"] = source
    for name, part in parts.items():
        path = output_dir / f"{name}.csv"
        part.to_csv(path, index=False, lineterminator="\n", encoding="utf-8")
        manifest["partitions"][name] = {
            "file": path.name,
            "rows": len(part),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "target_counts": {str(k): int(v) for k, v in part[TARGET].value_counts().sort_index().items()},
        }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output_dir = args.data_dir / "processed"
    # Invalida sucesso anterior inclusive se o download/leitura falhar.
    (output_dir / "manifest.json").unlink(missing_ok=True)
    workbook = acquire_data(args.data_dir / "raw")
    frame = normalize_data(pd.read_excel(workbook, header=1, engine="xlrd"))
    source = {
        "dataset": "UCI Default of Credit Card Clients (350)",
        "page_url": SOURCE_URL,
        "download_url": DOWNLOAD_URL,
        "doi": "10.24432/C55S3H",
        "citation": "Yeh, I. (2009). Default of Credit Card Clients. UCI Machine Learning Repository.",
        "license": "CC BY 4.0",
        "archive_sha256": ARCHIVE_SHA256,
        "workbook_sha256": hashlib.sha256(workbook.read_bytes()).hexdigest(),
        "target_mapping": {"default payment next month": TARGET},
    }
    # Valida as condicoes estruturais antes de emitir resultados da preparacao.
    manifest = write_partitions(frame, output_dir, args.seed, source=source)
    try:
        (output_dir / "profile.json").write_text(
            json.dumps(profile_data(frame), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except Exception:
        (output_dir / "manifest.json").unlink(missing_ok=True)
        raise
    print(json.dumps({"rows": len(frame), "partitions": {
        name: info["rows"] for name, info in manifest["partitions"].items()
    }, "manifest": str(output_dir / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
