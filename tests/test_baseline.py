import numpy as np
import pandas as pd
import pytest

from credit_monitor.baseline import FEATURES, calibration_without_seen_profiles
from credit_monitor.data import TARGET


@pytest.fixture
def reference():
    rng = np.random.default_rng(12)
    n = 80
    values = {"ID": np.arange(n), "LIMIT_BAL": rng.integers(1, 20, n) * 10000,
              "SEX": rng.integers(1, 3, n), "EDUCATION": rng.integers(0, 7, n),
              "MARRIAGE": rng.integers(0, 4, n), "AGE": rng.integers(19, 80, n)}
    for col in ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"):
        values[col] = rng.integers(-2, 9, n)
    for i in range(1, 7):
        values[f"BILL_AMT{i}"] = rng.integers(-1000, 10000, n)
        values[f"PAY_AMT{i}"] = rng.integers(0, 5000, n)
    values[TARGET] = np.tile([0, 0, 0, 1], n // 4)
    return pd.DataFrame(values, dtype="int64")


@pytest.fixture
def calibration(reference):
    frame = reference.iloc[:20].copy().reset_index(drop=True)
    frame["ID"] += 1000
    frame["LIMIT_BAL"] += 1000
    return frame


def test_calibration_excludes_seen_features_regardless_of_label_or_audit(reference, calibration):
    calibration.loc[0, list(FEATURES)] = reference.loc[0, list(FEATURES)]
    calibration.loc[0, TARGET] = 1
    calibration.loc[0, "SEX"] = 3 - reference.loc[0, "SEX"]
    before = calibration.copy(deep=True)
    clean, audit = calibration_without_seen_profiles(reference, calibration)
    assert len(clean) == 19
    assert 1000 not in set(clean.ID)
    assert audit == {"original_rows": 20, "excluded_seen_profiles": 1, "retained_rows": 19}
    pd.testing.assert_frame_equal(calibration, before)


def test_calibration_rejects_overlapping_ids(reference, calibration):
    calibration.loc[0, "ID"] = 0
    with pytest.raises(ValueError, match="ID"):
        calibration_without_seen_profiles(reference, calibration)


def test_calibration_rejects_when_no_unseen_rows_remain(reference):
    other = reference.copy()
    other["ID"] += 1000
    with pytest.raises(ValueError, match="calibracao"):
        calibration_without_seen_profiles(reference, other)
