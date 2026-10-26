"""Persistencia local de modelos e decisoes congeladas."""

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import pickle
import platform

import numpy as np
from sklearn.utils.validation import check_is_fitted

from credit_monitor.baseline import FEATURES, ModelPair


def _versions():
    return {"python": platform.python_version(), **{
        name: version(name) for name in ("numpy", "pandas", "scikit-learn")
    }}


def _check_thresholds(thresholds):
    if set(thresholds) != {"model", "dummy"}:
        raise ValueError("Limiares de model e dummy sao obrigatorios")
    if any(not isinstance(v, (int, float)) or not np.isfinite(v) or not 0 <= v <= 1
           for v in thresholds.values()):
        raise ValueError("Limiares invalidos")

def save_models(models, directory, *, thresholds, metadata):
    """Cria uma nova pasta; manifesto so aparece apos gravacao do modelo.

    Pickle serve apenas a artefatos locais confiaveis. Hash verifica integridade,
    nao autoria. Metadados do chamador devem conter somente agregados/proveniencia.
    """
    _check_thresholds(thresholds)
    check_is_fitted(models.model)
    check_is_fitted(models.dummy)
    payload = pickle.dumps(models, protocol=5)
    manifest = {
        "format_version": 1,
        "versions": _versions(),
        "features": list(FEATURES),
        "thresholds": thresholds,
        "model_sha256": hashlib.sha256(payload).hexdigest(),
        "metadata": metadata,
    }
    serialized = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "models.pkl").write_bytes(payload)
    (directory / "manifest.json").write_text(serialized, encoding="utf-8")
    return manifest


def load_models(directory, *, trusted=False):
    """Reabre apenas artefatos de origem confiavel, na mesma versao de runtime."""
    if trusted is not True:
        raise ValueError("Use trusted=True apenas para artefatos locais de origem confiavel")
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["format_version"] != 1 or manifest["versions"] != _versions():
        raise ValueError("Versoes de artefato/runtime incompativeis")
    if manifest["features"] != list(FEATURES):
        raise ValueError("Features do artefato divergem do contrato do modelo")
    _check_thresholds(manifest["thresholds"])
    payload = (directory / "models.pkl").read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest["model_sha256"]:
        raise ValueError("SHA-256 do modelo divergente")
    models = pickle.loads(payload)
    if not isinstance(models, ModelPair):
        raise ValueError("Tipo de artefato inesperado")
    check_is_fitted(models.model)
    check_is_fitted(models.dummy)
    return models, manifest
