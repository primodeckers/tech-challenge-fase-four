"""Contratos de ingestao: sem imputacao, coercao ou descarte de linhas."""

import pandera.pandas as pa

from credit_monitor.data import TARGET


def inference_schema() -> pa.DataFrameSchema:
    """Lote UCI sem rotulo; todas as colunas de ingestao permanecem obrigatorias."""
    columns = {
        "ID": pa.Column("int64", unique=True),
        "LIMIT_BAL": pa.Column("int64", pa.Check.gt(0, name="positive_credit_limit")),
        "AGE": pa.Column("int64", pa.Check.in_range(19, 100, name="adult_age_range")),
        "SEX": pa.Column("int64", pa.Check.isin([1, 2], name="sex_code")),
        "EDUCATION": pa.Column("int64", pa.Check.isin(range(7), name="education_code")),
        "MARRIAGE": pa.Column("int64", pa.Check.isin(range(4), name="marriage_code")),
    }
    for name in ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"):
        columns[name] = pa.Column("int64", pa.Check.in_range(-2, 9, name="repayment_code"))
    for i in range(1, 7):
        columns[f"BILL_AMT{i}"] = pa.Column("int64")  # Faturas negativas existem na fonte.
        columns[f"PAY_AMT{i}"] = pa.Column("int64", pa.Check.ge(0, name="nonnegative_payment"))
    return pa.DataFrameSchema(
        columns, checks=pa.Check(lambda frame: len(frame) > 0, name="nonempty_batch"),
        strict=True, coerce=False, unique_column_names=True, name="credit_inference_v1",
    )


def training_schema() -> pa.DataFrameSchema:
    """Mesmo contrato de ingestao, acrescido do rotulo binario obrigatorio."""
    schema = inference_schema().add_columns({
        TARGET: pa.Column("int64", pa.Check.isin([0, 1], name="binary_target")),
    })
    schema.name = "credit_training_v1"
    return schema
