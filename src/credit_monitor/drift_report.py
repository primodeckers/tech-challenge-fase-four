"""Adaptador Evidently 0.7.23 para a politica de drift do projeto."""

from pathlib import Path
from threading import Lock
from html import escape

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


def write_overview(comparisons, output_html):
    """Resumo de unidades/limiares e ligacoes aos dois HTMLs nativos."""
    sections = []
    def number(value):
        return "indisponível" if value is None else f"{value:.6g}"
    for scenario, result in comparisons.items():
        rows = []
        for column, info in result["features"].items():
            decision = "ALERTA" if info["alert"] is True else "sem alerta" if info["alert"] is False else "indisponível"
            values = [escape(column), number(info["psi"]), number(info["psi_threshold"]),
                      number(info["ks"]["statistic"]), number(info["ks_threshold"]),
                      number(info["ks_pvalue_holm"]), escape(info["status"]), decision]
            rows.append(f'<tr class="{"alert" if info["alert"] else ""}">' + ''.join(f'<td>{v}</td>' for v in values) + '</tr>')
        title = "Controle" if scenario == "control" else "Data Drift simulado"
        sections.append(f'<section><h2>{title}</h2><p>Referência: {result["n_reference"]:,} registros · '
            f'Lote: {result["n_current"]:,} · Alertas: {len(result["alerted_features"])} / {len(FEATURES)} · '
            f'<a href="{scenario}.html">Abrir Evidently</a></p><div class="scroll"><table>'
            '<thead><tr><th>Feature</th><th>PSI</th><th>Limite PSI</th><th>KS D</th><th>Limite D</th>'
            '<th>p ajustado (Holm)</th><th>Estado</th><th>Decisão</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div></section>')
    html = '<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">' \
        '<title>S05 · Monitoramento de dados</title><style>body{font:15px/1.6 system-ui,sans-serif;background:#0e1726;color:#e4eaf3;margin:0;padding:32px}' \
        'main{max-width:1400px;margin:auto}h1{font-size:32px;margin-bottom:8px}h2{margin-top:0}a{color:#f18bb3}section{background:#162338;padding:24px;margin:24px 0;border:1px solid #34445b;border-radius:12px}' \
        'table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:10px 12px;text-align:left;border-bottom:1px solid #34445b;white-space:nowrap}th{color:#a9b9d0}' \
        '.alert{background:#42283c}.scroll{overflow:auto}.note{color:#afc0d7;max-width:1100px}.tag{color:#f18bb3;font-size:12px;letter-spacing:2px}</style><main>' \
        '<p class="tag">TECH CHALLENGE / FASE 04</p><h1>Data Drift · PSI + KS</h1>' \
        '<p class="note">Bins fixados na referência; envelopes definidos por bootstrap da calibração antes da produção. '
    html += 'Alerta exige PSI e KS D acima dos respectivos limites e p ajustado ≤ 0,05. Mudança de suporte constante é sinalizada separadamente.</p>' \
        '<p class="note">PSI e KS D são adimensionais (D entre 0 e 1). p-valores KS são aproximações contínuas em dados com empates; '
    html += 'Holm corrige as 20 features por comparação, não o monitoramento repetido no tempo. p não é a probabilidade de uma hipótese ser verdadeira. '
    html += 'Os envelopes são critérios operacionais, sem garantia de taxa global de falso alerta. Sem alerta não comprova estabilidade. Não é teste de Concept Drift.</p>'
    html += ''.join(sections) + '</main></html>'
    Path(output_html).write_text(html, encoding="utf-8")
