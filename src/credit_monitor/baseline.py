"""Baseline de credito: features explicitas e separacao de avaliacao."""

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from credit_monitor.contracts import validate_batch
from credit_monitor.data import TARGET


@dataclass(frozen=True)
class ModelPair:
    model: Pipeline
    dummy: DummyClassifier


def evaluate_binary(labels, scores, threshold):
    """Metricas agregadas; linhas da matriz=situacao real, colunas=predicao, ordem0/1."""
    y, p = _validated_arrays(labels, scores)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Limiar precisa estar entre 0 e 1")
    predicted = (p >= threshold).astype(int)
    both_classes = len(np.unique(y)) == 2
    return {
        "rows": len(y), "threshold": float(threshold),
        "prevalence": float(y.mean()), "predicted_positive_rate": float(predicted.mean()),
        "roc_auc": float(roc_auc_score(y, p)) if both_classes else None,
        "average_precision": float(average_precision_score(y, p)) if both_classes else None,
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "brier": float(brier_score_loss(y, p, pos_label=1)),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }


def select_threshold(labels, scores):
    """Maximiza F1 na grade fixa 0.05..0.95; desempate pelo maior limiar.

    Deve receber apenas calibracao. Nao otimiza custos financeiros nem probabilidades.
    """
    y, p = _validated_arrays(labels, scores)
    if len(np.unique(y)) != 2:
        raise ValueError("Selecao do limiar requer as duas classes")
    candidates = []
    for threshold in np.arange(5, 96) / 100:
        predicted = p >= threshold
        tp = int(((y == 1) & predicted).sum())
        fp = int(((y == 0) & predicted).sum())
        fn = int(((y == 1) & ~predicted).sum())
        score = 2 * tp / (2 * tp + fp + fn)
        candidates.append((score, float(threshold)))
    return max(candidates)[1]


def _validated_arrays(labels, scores):
    y, p = np.asarray(labels), np.asarray(scores, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) == 0 or len(y) != len(p):
        raise ValueError("Rotulos e scores precisam de vetores nao vazios com mesmo tamanho")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Rotulos binarios e probabilidades finitas entre0 e1 sao obrigatorios")
    return y.astype(int), p


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
