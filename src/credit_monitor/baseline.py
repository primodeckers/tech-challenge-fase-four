"""Baseline de credito: features explicitas e separacao de avaliacao."""

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from credit_monitor.contracts import validate_batch
from credit_monitor.data import TARGET


@dataclass(frozen=True)
class ModelPair:
    model: Pipeline
    dummy: DummyClassifier


def train_models(reference: pd.DataFrame) -> ModelPair:
    """Ajusta scaler, regressao logistica e dummy apenas nos dados de referencia."""
    reference = validate_batch(reference, mode="training")
    if set(reference[TARGET].unique()) != {0, 1}:
        raise ValueError("Treino requer as duas classes binarias")
    x, y = reference.loc[:, FEATURES], reference[TARGET]
    model = Pipeline([("scale", StandardScaler()),
                      ("classifier", LogisticRegression(random_state=42, max_iter=2000, solver="lbfgs"))])
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x, y)
    dummy = DummyClassifier(strategy="prior").fit(x, y)
    return ModelPair(model, dummy)


def predict_scores(model, batch: pd.DataFrame) -> np.ndarray:
    """Probabilidade da classe 1; exige lote sem alvo e nunca reajusta o modelo."""
    batch = validate_batch(batch, mode="inference")
    if list(model.classes_) != [0, 1]:
        raise ValueError("Modelo precisa das classes 0 e 1 nessa ordem")
    scores = np.asarray(model.predict_proba(batch.loc[:, FEATURES]))[:, 1]
    if scores.shape != (len(batch),) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Probabilidades invalidas retornadas pelo modelo")
    return scores

FEATURES = ("LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
            *(f"BILL_AMT{i}" for i in range(1, 7)), *(f"PAY_AMT{i}" for i in range(1, 7)))


def calibration_without_seen_profiles(reference, calibration):
    """Exclui perfis X vistos no treino, sem consultar rotulos ou alterar CSVs."""
    reference = validate_batch(reference, mode="training")
    calibration = validate_batch(calibration, mode="training")
    if set(reference.ID) & set(calibration.ID):
        raise ValueError("IDs de referencia e calibracao devem ser disjuntos")
    known = pd.MultiIndex.from_frame(reference.loc[:, FEATURES])
    seen = pd.MultiIndex.from_frame(calibration.loc[:, FEATURES]).isin(known)
    clean = calibration.loc[~seen].copy()
    if clean.empty:
        raise ValueError("Nenhum perfil independente disponivel na calibracao")
    return clean, {"original_rows": len(calibration), "excluded_seen_profiles": int(seen.sum()),
                   "retained_rows": len(clean)}
