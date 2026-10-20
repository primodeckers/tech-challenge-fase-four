import pandas as pd
import pandera.pandas as pa
import pytest

from credit_monitor.contracts import InvalidBatch, execute_validated, inference_schema, training_schema, validate_batch


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


def test_validation_reports_multiple_rules_with_counts(batch):
    batch.loc[0, "AGE"] = 18
    batch.loc[1, "ID"] = 101
    batch.loc[2, "LIMIT_BAL"] = 0
    with pytest.raises(InvalidBatch) as caught:
        validate_batch(batch, mode="training")
    violations = caught.value.as_dict()["violations"]
    assert { (v["column"], v["rule"]): v["count"] for v in violations } == {
        ("AGE", "adult_age_range"): 1,
        ("ID", "unique_id"): 2,
        ("LIMIT_BAL", "positive_credit_limit"): 1,
    }


def test_validation_hides_values_unknown_column_names_and_pandera_context(batch, capsys):
    secret = "PRIVATE_CUSTOMER_987654"
    batch["AGE"] = [secret, secret, secret]
    batch[secret] = secret
    with pytest.raises(InvalidBatch) as caught:
        validate_batch(batch, mode="training")
    error = caught.value
    assert error.as_dict()["status"] == "rejected"
    assert secret not in str(error)
    assert secret not in repr(error.__dict__)
    assert error.__context__ is None
    assert error.__cause__ is None
    assert capsys.readouterr().out == ""


def test_validation_missing_column_identifies_only_known_name(batch):
    with pytest.raises(InvalidBatch) as caught:
        validate_batch(batch.drop(columns="AGE"), mode="training")
    assert {"column": "AGE", "rule": "missing_column", "count": 1} in caught.value.as_dict()["violations"]


@pytest.mark.parametrize("mode", ["training", "inference"])
def test_validation_returns_unchanged_valid_batch(batch, mode):
    if mode == "inference":
        batch = batch.drop(columns="default_next_month")
    pd.testing.assert_frame_equal(validate_batch(batch, mode=mode), batch)


def test_validation_rejects_wrong_input_and_unknown_mode(batch):
    with pytest.raises(InvalidBatch):
        validate_batch([1, 2, 3])
    with pytest.raises(ValueError, match="mode"):
        validate_batch(batch, mode="typo")


def test_global_pandera_disable_cannot_bypass_public_validation(batch):
    from pandera.config import config_context

    batch.loc[0, "AGE"] = 18
    with config_context(validation_enabled=False):
        with pytest.raises(InvalidBatch):
            validate_batch(batch, mode="training")


@pytest.mark.parametrize("case", ["null", "duplicate", "range", "missing", "type", "empty", "target_present"])
def test_invalid_batch_never_reaches_downstream_operation(batch, case):
    frame = batch.drop(columns="default_next_month")
    if case == "null":
        frame["AGE"] = frame.AGE.astype("Int64")
        frame.loc[0, "AGE"] = pd.NA
    elif case == "duplicate":
        frame.loc[1, "ID"] = 101
    elif case == "range":
        frame.loc[0, "LIMIT_BAL"] = 0
    elif case == "missing":
        frame = frame.drop(columns="BILL_AMT1")
    elif case == "type":
        frame["LIMIT_BAL"] = frame.LIMIT_BAL.astype(str)
    elif case == "empty":
        frame = frame.iloc[:0]
    else:
        frame["default_next_month"] = batch.default_next_month
    consumed = []

    def operation(validated):
        consumed.append(validated)
        return "must not run"

    with pytest.raises(InvalidBatch):
        execute_validated(frame, operation)
    assert consumed == []


@pytest.mark.parametrize("mode", ["training", "inference"])
def test_validated_operation_receives_full_batch_once_and_returns_result(batch, mode):
    if mode == "inference":
        batch = batch.drop(columns="default_next_month")
    before = batch.copy(deep=True)
    consumed = []

    def operation(validated):
        consumed.append(validated.copy(deep=True))
        validated.loc[0, "AGE"] = 50
        return {"processed_rows": len(validated)}

    assert execute_validated(batch, operation, mode=mode) == {"processed_rows": 3}
    assert len(consumed) == 1
    pd.testing.assert_frame_equal(consumed[0], before)
    pd.testing.assert_frame_equal(batch, before)


def test_downstream_failure_propagates_without_becoming_validation_success(batch):
    failure = RuntimeError("synthetic downstream failure")

    def operation(validated):
        raise failure

    with pytest.raises(RuntimeError) as caught:
        execute_validated(batch, operation, mode="training")
    assert caught.value is failure


def test_multilevel_columns_are_rejected_with_sanitized_diagnostics(batch):
    secret = "PRIVATE_COLUMN_GROUP_876543"
    batch.columns = pd.MultiIndex.from_product([[secret], batch.columns])
    consumed = []
    with pytest.raises(InvalidBatch) as caught:
        execute_validated(batch, lambda valid: consumed.append(valid), mode="training")
    assert caught.value.as_dict() == {
        "status": "rejected",
        "violations": [{"column": None, "rule": "flat_columns_required", "count": 1}],
    }
    assert secret not in str(caught.value)
    assert caught.value.__context__ is None
    assert consumed == []


def test_nested_invalid_values_count_one_dtype_event(batch):
    batch["AGE"] = [{"private": 1}, {"private": 2}, {"private": 3}]
    with pytest.raises(InvalidBatch) as caught:
        validate_batch(batch, mode="training")
    failures = caught.value.as_dict()["violations"]
    assert {"column": "AGE", "rule": "integer_dtype", "count": 1} in failures
    assert "private" not in str(caught.value)


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
