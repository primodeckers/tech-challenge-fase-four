"""Controle e simulacoes reproduziveis de distribuicao e conceito."""

import numpy as np
import pandas as pd

from credit_monitor.contracts import validate_batch
from credit_monitor.baseline import predict_scores
from credit_monitor.data import TARGET

TRANSFORMATIONS = {"LIMIT_BAL": (3, 2), "PAY_AMT1": (1, 2)}


def build_concept_scenario(production, *, seed=42):
    """Troca Y entre dois grupos de PAY_0, preservando X e prevalencia.

    Sorteia 1/4 do menor pool em cada direcao, sem consultar o modelo.
    Candidatos ordenados por ID tornam a selecao independente da ordem do lote.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("Seed precisa ser inteiro nao negativo")
    control = copy_control(production)
    if TARGET not in control:
        raise ValueError("Geracao de Concept Drift requer rotulos")
    up = control.loc[(control.PAY_0 <= 0) & (control[TARGET] == 0), 'ID'].sort_values().to_numpy()
    down = control.loc[(control.PAY_0 > 0) & (control[TARGET] == 1), 'ID'].sort_values().to_numpy()
    count = min(len(up), len(down)) // 4
    if count == 0:
        raise ValueError("Amostra insuficiente para intervencao efetiva nos dois grupos")
    rng = np.random.Generator(np.random.PCG64(seed))
    control.loc[control.ID.isin(rng.choice(up, count, replace=False)), TARGET] = 1
    control.loc[control.ID.isin(rng.choice(down, count, replace=False)), TARGET] = 0
    return copy_control(control)


def predict_scenarios(model, threshold, scenarios):
    """Prediz ambos os cenarios com o mesmo estimador e limiar, sem fit."""
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Limiar precisa ser finito entre0 e1")
    if set(scenarios) != {"control", "data_drift"}:
        raise ValueError("Controle e data_drift sao obrigatorios")
    # Validar o par inteiro antes de fazer qualquer inferencia.
    batches = {name: copy_control(scenarios[name]) for name in ("control", "data_drift")}
    control, drift = batches["control"], batches["data_drift"]
    if control.ID.tolist() != drift.ID.tolist() or set(control.columns) != set(drift.columns):
        raise ValueError("Cenarios precisam dos mesmos IDs, ordem e colunas")
    if TARGET in control and control[TARGET].tolist() != drift[TARGET].tolist():
        raise ValueError("Rotulos historicos precisam ser preservados entre cenarios")
    results = {}
    for name, frame in batches.items():
        scores = predict_scores(model, frame.drop(columns=TARGET) if TARGET in frame else frame)
        results[name] = pd.DataFrame({"ID": frame.ID.to_numpy(copy=True),
                                      "probability_default": scores,
                                      "prediction": (scores >= threshold).astype("int64")})
    return results

def build_scenarios(production):
    """Dois cenarios pareados: controle intacto e duas features escaladas.

    Aritmetica inteira com floor; sem RNG, clipping, retreino ou alteracao de Y.
    """
    control = copy_control(production)
    drift = control.copy(deep=True)
    for column, (numerator, denominator) in TRANSFORMATIONS.items():
        # int Python antes da multiplicacao evita overflow silencioso do numpy.
        values = [int(value) * numerator // denominator for value in control[column]]
        if any(value > np.iinfo(np.int64).max for value in values):
            raise ValueError(f"Transformacao de {column} excede int64")
        drift[column] = np.asarray(values, dtype=np.int64)
        if drift[column].equals(control[column]):
            raise ValueError(f"Transformacao nao alterou a feature {column}")
    return {"control": control, "data_drift": copy_control(drift)}


def copy_control(production: pd.DataFrame) -> pd.DataFrame:
    """Copia validada, sem modificar fonte, valores, ordem, IDs ou rotulos."""
    mode = "training" if isinstance(production, pd.DataFrame) and TARGET in production else "inference"
    return validate_batch(production, mode=mode).copy(deep=True)
