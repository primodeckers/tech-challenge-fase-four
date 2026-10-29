"""Adaptador Evidently 0.7.23 para a politica de drift do projeto."""

from pathlib import Path
from threading import Lock

from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset
from evidently.legacy.calculations.stattests.registry import StatTest, register_stattest
from evidently.legacy.core import ColumnType

from credit_monitor.baseline import FEATURES
from credit_monitor.drift import compare_drift

_REPORT_LOCK = Lock()
_METHOD = "credit_psi_ks_holm_v1"

def export_evidently(reference, current, policy, output_html):
    """HTML nativo com os MESMOS alertas do JSON; score exibido e KS D.

    O teste registrado adapta a decisao composta ja calculada, em vez de deixar
    defaults do Evidently produzirem outra politica. Registro global protegido
    por lock durante o calculo/render. Nenhum ID, alvo ou atributo de auditoria
    e enviado ao Dataset do Evidently.
    """
    output_html = Path(output_html)
    if output_html.exists():
        raise FileExistsError("Relatorio existente nao sera sobrescrito")
    comparison = compare_drift(reference, current, policy)
    if comparison["status"] != "ok":
        raise ValueError("Comparacao indisponivel para emitir relatorio de sucesso")

    def calibrated_decision(reference_column, current_column, feature_type, threshold):
        info = comparison["features"][current_column.name]
        return float(info["ks"]["statistic"]), bool(info["alert"])

    definition = DataDefinition(numerical_columns=list(FEATURES))
    current_dataset = Dataset.from_pandas(current.loc[:, FEATURES].copy(), data_definition=definition)
    reference_dataset = Dataset.from_pandas(reference.loc[:, FEATURES].copy(), data_definition=definition)
    limits = {column: policy["features"][column]["ks_threshold"] or 0.0 for column in FEATURES}
    with _REPORT_LOCK:
        register_stattest(StatTest(name=_METHOD, display_name="KS D + PSI (Holm)",
            allowed_feature_types=[ColumnType.Numerical]), calibrated_decision)
        snapshot = Report([DataDriftPreset(columns=list(FEATURES), method=_METHOD,
            per_column_threshold=limits, drift_share=1 / len(FEATURES), include_tests=False)],
            metadata={"policy": "PSI and KS D calibration envelopes plus Holm-adjusted p; constant-support changes separate",
                      "score_unit": "KS D (0..1); not a p-value"}).run(
                current_data=current_dataset, reference_data=reference_dataset)
        output_html.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(output_html))
        result = snapshot.dict()
    return {"comparison": comparison, "evidently": result}
