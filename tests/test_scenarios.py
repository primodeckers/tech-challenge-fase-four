import numpy as np
import pandas as pd
import pytest

from credit_monitor.contracts import InvalidBatch, validate_batch
from credit_monitor.data import TARGET
from credit_monitor.scenarios import copy_control


@pytest.fixture
def production():
    values = {"ID": [1001, 1002, 1003, 1004], "LIMIT_BAL": [10000, 20001, 30000, 40001],
              "SEX": [1, 2, 1, 2], "EDUCATION": [0, 5, 6, 1], "MARRIAGE": [0, 1, 3, 2],
              "AGE": [19, 30, 50, 100]}
    for col in ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"):
        values[col] = [-2, -1, 0, 8]
    for i in range(1, 7):
        values[f"BILL_AMT{i}"] = [-100, 0, 1000, 5000]
        values[f"PAY_AMT{i}"] = [0, 1, 1001, 2000]
    values[TARGET] = [0, 1, 0, 1]
    return pd.DataFrame(values, dtype="int64")


@pytest.mark.parametrize("labeled", [True, False])
def test_control_is_identical_and_independent_of_input(production, labeled):
    if not labeled:
        production = production.drop(columns=TARGET)
    before = production.copy(deep=True)
    control = copy_control(production)
    pd.testing.assert_frame_equal(control, before)
    control.loc[0, "LIMIT_BAL"] = 99999
    pd.testing.assert_frame_equal(production, before)


def test_control_rejects_invalid_production_instead_of_repairing(production):
    production.loc[0, "LIMIT_BAL"] = 0
    with pytest.raises(InvalidBatch):
        copy_control(production)
    assert production.loc[0, "LIMIT_BAL"] == 0
