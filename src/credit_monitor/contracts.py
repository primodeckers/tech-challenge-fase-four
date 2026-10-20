"""Contratos de ingestao: sem imputacao, coercao ou descarte de linhas."""

from collections import Counter
from dataclasses import asdict, dataclass
import json
from typing import Callable, Literal, TypeVar

import pandas as pd
import pandera.pandas as pa
from pandera.config import ValidationDepth, config_context

from credit_monitor.data import TARGET


Result = TypeVar("Result")


def execute_validated(
    frame: pd.DataFrame, operation: Callable[[pd.DataFrame], Result], *,
    mode: Literal["training", "inference"] = "inference",
) -> Result:
    """Barreira para consumidores futuros: nenhuma operacao recebe lote invalido.

    O consumidor e responsavel por selecionar features; isto nao treina nem prediz.
    Falhas do consumidor sao propagadas e nunca convertidas em validacao bem-sucedida.
    """
    validated = validate_batch(frame, mode=mode)
    return operation(validated)


@dataclass(frozen=True)
class Violation:
    column: str | None
    rule: str
    count: int


class InvalidBatch(ValueError):
    """Rejeicao agregada; nao retém failure_cases ou excecoes internas com dados."""

    def __init__(self, violations: tuple[Violation, ...]):
        self.violations = violations
        super().__init__(json.dumps(self.as_dict(), sort_keys=True))

    def as_dict(self) -> dict:
        return {"status": "rejected", "violations": [asdict(v) for v in self.violations]}


def validate_batch(
    frame: pd.DataFrame, *, mode: Literal["training", "inference"] = "inference",
) -> pd.DataFrame:
    """Valida o lote inteiro; falhas nunca retornam dados parcialmente aceitos.

    Contagens sao ocorrencias por regra, nao total de linhas distintas invalidas.
    Falhas estruturais/tipo contam um evento; IDs duplicados contam todas as linhas
    participantes. Nomes de colunas inesperadas e valores nunca entram no resumo.
    """
    if mode not in ("training", "inference"):
        raise ValueError("mode deve ser training ou inference")
    if not isinstance(frame, pd.DataFrame):
        raise InvalidBatch((Violation(None, "dataframe_required", 1),))
    if isinstance(frame.columns, pd.MultiIndex):
        raise InvalidBatch((Violation(None, "flat_columns_required", 1),))
    schema = training_schema() if mode == "training" else inference_schema()
    violations = ()
    with config_context(validation_enabled=True, validation_depth=ValidationDepth.SCHEMA_AND_DATA):
        try:
            return schema.validate(frame, lazy=True)
        except (pa.errors.SchemaErrors, pa.errors.SchemaError) as error:
            violations = _summarize_errors(error, schema)
    # Fora do except: nem o encadeamento implicito retem dados da excecao Pandera.
    raise InvalidBatch(violations)


def _summarize_errors(error, schema: pa.DataFrameSchema) -> tuple[Violation, ...]:
    reason_names = {
        "COLUMN_NOT_IN_SCHEMA": "unexpected_column",
        "COLUMN_NOT_IN_DATAFRAME": "missing_column",
        "DUPLICATE_COLUMN_LABELS": "duplicate_column",
        "SERIES_CONTAINS_DUPLICATES": "unique_id",
        "SERIES_CONTAINS_NULLS": "not_null",
        "WRONG_DATATYPE": "integer_dtype",
    }
    check_names = {
        "positive_credit_limit", "adult_age_range", "sex_code", "education_code",
        "marriage_code", "repayment_code", "nonnegative_payment", "binary_target",
        "nonempty_batch",
    }
    counts = Counter()
    for failure in getattr(error, "schema_errors", [error]):
        reason = failure.reason_code.name
        column = failure.column_name
        if reason == "COLUMN_NOT_IN_DATAFRAME":
            column = failure.failure_cases
        if not isinstance(column, str) or column not in schema.columns:
            column = None
        check = getattr(failure.check, "name", None)
        rule = reason_names.get(reason, check if check in check_names else "schema_violation")
        cases = failure.failure_cases
        structural = reason in {
            "COLUMN_NOT_IN_SCHEMA", "COLUMN_NOT_IN_DATAFRAME",
            "DUPLICATE_COLUMN_LABELS", "WRONG_DATATYPE",
        }
        count = len(cases) if not structural and isinstance(cases, (pd.DataFrame, pd.Series)) else 1
        counts[(column, rule)] += max(count, 1)
    return tuple(Violation(column, rule, count) for (column, rule), count in
                 sorted(counts.items(), key=lambda item: (item[0][0] or "", item[0][1])))


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
