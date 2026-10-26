"""Controle e simulacoes deterministicas de Data Drift."""

import pandas as pd

from credit_monitor.contracts import validate_batch
from credit_monitor.data import TARGET


def copy_control(production: pd.DataFrame) -> pd.DataFrame:
    """Copia validada, sem modificar fonte, valores, ordem, IDs ou rotulos."""
    mode = "training" if isinstance(production, pd.DataFrame) and TARGET in production else "inference"
    return validate_batch(production, mode=mode).copy(deep=True)
