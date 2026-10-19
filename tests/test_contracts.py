import pandas as pd
import pandera.pandas as pa
import pytest

from credit_monitor.contracts import inference_schema, training_schema


@pytest.fixture
def batch():
    values = {"ID": [101, 102, 103], "LIMIT_BAL": [10000, 20000, 30000],
              "SEX": [1, 2, 1], "EDUCATION": [0, 5, 6], "MARRIAGE": [0, 1, 3],
              "AGE": [19, 40, 100]}
    for name in ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"):
        values[name] = [-2, -1, 9]
    for i in range(1, 7):
        values[f"BILL_AMT{i}"] = [-100, 0, 1000]
        values[f"PAY_AMT{i}"] = [0, 50, 100]
    values["default_next_month"] = [0, 1, 0]
    return pd.DataFrame(values, dtype="int64")


def test_training_preserves_valid_data_and_special_codes(batch):
    original = batch.copy(deep=True)
    result = training_schema().validate(batch, lazy=True)
    pd.testing.assert_frame_equal(result, original)
    pd.testing.assert_frame_equal(batch, original)


def test_inference_requires_no_target_and_training_requires_target(batch):
    unlabeled = batch.drop(columns="default_next_month")
    pd.testing.assert_frame_equal(inference_schema().validate(unlabeled), unlabeled)
    with pytest.raises(pa.errors.SchemaErrors):
        training_schema().validate(unlabeled, lazy=True)
    with pytest.raises(pa.errors.SchemaErrors):
        inference_schema().validate(batch, lazy=True)


@pytest.mark.parametrize("column,value", [
    ("AGE", 18), ("AGE", 101), ("LIMIT_BAL", 0), ("LIMIT_BAL", -1),
    ("SEX", 3), ("EDUCATION", 7), ("MARRIAGE", 4),
    ("PAY_0", -3), ("PAY_6", 10), ("PAY_AMT1", -1), ("default_next_month", 2),
])
def test_training_rejects_invalid_domain_without_mutation(batch, column, value):
    batch.loc[0, column] = value
    before = batch.copy(deep=True)
    with pytest.raises(pa.errors.SchemaErrors):
        training_schema().validate(batch, lazy=True)
    pd.testing.assert_frame_equal(batch, before)


@pytest.mark.parametrize("case", ["null", "duplicate_id", "missing", "extra", "string", "fraction", "empty", "duplicate_column"])
@pytest.mark.parametrize("mode", ["training", "inference"])
def test_both_schemas_reject_structural_violations(batch, case, mode):
    if mode == "inference":
        batch = batch.drop(columns="default_next_month")
    if case == "null":
        batch["LIMIT_BAL"] = batch.LIMIT_BAL.astype("Int64")
        batch.loc[0, "LIMIT_BAL"] = pd.NA
    elif case == "duplicate_id":
        batch.loc[1, "ID"] = 101
    elif case == "missing":
        batch = batch.drop(columns="AGE")
    elif case == "extra":
        batch["unexpected"] = 1
    elif case == "string":
        batch["AGE"] = batch.AGE.astype(str)
    elif case == "fraction":
        batch["AGE"] = [19.5, 40.0, 100.0]
    elif case == "empty":
        batch = batch.iloc[:0]
    else:
        batch = pd.concat([batch, batch[["AGE"]]], axis=1)
    before = batch.copy(deep=True)
    schema = training_schema() if mode == "training" else inference_schema()
    with pytest.raises(pa.errors.SchemaErrors):
        schema.validate(batch, lazy=True)
    pd.testing.assert_frame_equal(batch, before)
