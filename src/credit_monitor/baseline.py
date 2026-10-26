"""Baseline de credito: features explicitas e separacao de avaliacao."""

import pandas as pd

from credit_monitor.contracts import validate_batch

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
