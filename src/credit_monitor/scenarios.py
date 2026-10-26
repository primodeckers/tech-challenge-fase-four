"""Controle e simulacoes deterministicas de Data Drift."""

import numpy as np
import pandas as pd

from credit_monitor.contracts import validate_batch
from credit_monitor.data import TARGET

TRANSFORMATIONS = {"LIMIT_BAL": (3, 2), "PAY_AMT1": (1, 2)}

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
