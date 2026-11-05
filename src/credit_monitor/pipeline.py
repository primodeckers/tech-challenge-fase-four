"""Execuções independentes dos cenários, do contrato ao relatório de drift."""

import hashlib
from io import BytesIO
import json
from pathlib import Path
from uuid import uuid4

import pandas as pd

from credit_monitor.baseline import evaluate_binary, predict_scores
from credit_monitor.concept import predict_concept
from credit_monitor.contracts import InvalidBatch, Violation, validate_batch
from credit_monitor.data import TARGET
from credit_monitor.drift import compare_drift
from credit_monitor.drift_report import render_evidently
from credit_monitor.model_store import load_models
from credit_monitor.scenario_run import _verified_partition
from credit_monitor.scenarios import build_concept_scenario, build_scenarios
from credit_monitor.telemetry import RunLog, prediction_summary

SCENARIOS = ('control', 'data_drift', 'concept_drift')
FAILURE_ACTIONS = {
    'ingestion': 'Verificar origem, integridade e formato do CSV antes de reenviar.',
    'validation': 'Revisar o contrato de ingestão e a versão das dependências.',
    'transformation': 'Revisar os pré-requisitos e domínios da simulação.',
    'prediction': 'Conferir o modelo local, sua origem e a versão do ambiente.',
    'drift': 'Conferir a política S05, o manifesto e a referência de calibração.',
    'report': 'Conferir o espaço em disco, permissões e ambiente do Evidently.',
}
REPORT_FILES = ('reports/report.json', 'reports/drift.html')


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _read_batch(data_dir, batch_path, manifest):
    official = batch_path is None
    if official:
        partition = manifest['partitions']['production']
        if partition['file'] != 'production.csv':
            raise ValueError('Nome de partição inesperado')
        batch_path = data_dir / 'production.csv'
    payload = Path(batch_path).read_bytes()
    digest = _sha(payload)
    if official and digest != partition['sha256']:
        raise ValueError('SHA-256 de produção divergente')
    frame = pd.read_csv(BytesIO(payload))
    if not isinstance(frame.index, pd.RangeIndex):
        raise ValueError('CSV com campos adicionais fora do cabeçalho')
    if official and len(frame) != partition['rows']:
        raise ValueError('Contagem de produção divergente')
    return frame, digest


def _validate(frame):
    mode = 'training' if TARGET in frame else 'inference'
    return validate_batch(frame, mode=mode)


def _transform(frame, scenario):
    if scenario == 'data_drift':
        return build_scenarios(frame)['data_drift']
    if scenario == 'concept_drift':
        return build_concept_scenario(frame)
    return frame


def _predict(model, threshold, control, current, scenario):
    if scenario == 'concept_drift':
        predictions, performance = predict_concept(model, threshold, control, current)
        return predictions.probability_default, performance
    labeled = TARGET in current
    scores = predict_scores(model, current.drop(columns=TARGET) if labeled else current)
    if not labeled:
        performance = {'status': 'unavailable_without_labels'}
    elif scenario == 'data_drift':
        performance = {'status': 'unavailable_under_feature_intervention'}
    else:
        performance = {'status': 'available', 'metrics': evaluate_binary(current[TARGET], scores, threshold)}
    return scores, performance


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n',
                    encoding='utf-8')


def _read_policy(policy_dir, source):
    manifest = json.loads((policy_dir / 'manifest.json').read_bytes())
    if (manifest['stage'] != 'S05' or manifest['format_version'] != 1
            or manifest['data_manifest_sha256'] != source['data_manifest_sha256']):
        raise ValueError('Política não corresponde à origem dos dados')
    payload = (policy_dir / 'policy.json').read_bytes()
    digest = _sha(payload)
    if manifest['files']['policy.json']['sha256'] != digest:
        raise ValueError('SHA-256 da política divergente')
    source['policy_sha256'] = digest
    return json.loads(payload)


def _emit_drift_alerts(log, comparison):
    for name in comparison['alerted_features']:
        feature = comparison['features'][name]
        value = {'psi': feature['psi'], 'ks_d': feature['ks']['statistic'],
                 'p_holm': feature['ks_pvalue_holm']}
        threshold = {'psi': feature['psi_threshold'], 'ks_d': feature['ks_threshold'],
                     'p_holm': comparison['alpha']}
        if feature['reason'] == 'constant_support_changed':
            value['support_changed'] = True
            threshold = {'support_changed': False}
        log.alert(kind='data_drift', severity='warning', rule=feature['reason'],
                  feature=name, value=value, threshold=threshold,
                  action='Investigar origem, distribuição e impacto antes de qualquer ajuste no modelo.')


def _finish(log, status, rows, source, *, reason=None, files=()):
    directory = log.path.parent
    result = {'stage': 'S07', 'format_version': 1, 'status': status, 'run_id': log.run_id,
              'scenario': log.scenario, 'rows': rows, 'source': source,
              'failed_stage': log.current_stage if status != 'succeeded' else None,
              'reason': reason, 'alert_count': log.alert_count,
              'files': {name: {'sha256': _sha((directory / name).read_bytes())} for name in files}}
    temporary = directory / '.result.tmp'
    _write_json(temporary, result)
    temporary.replace(directory / 'result.json')
    return result


def run_pipeline(*, data_dir, model_dir, policy_dir, runs_dir, scenario='control', batch_path=None,
                 trusted_model=False):
    """Retorna estado terminal e mantém o histórico local em uma pasta nova."""
    if scenario not in SCENARIOS:
        raise ValueError('Cenário desconhecido')
    data_dir, model_dir, policy_dir, runs_dir = map(Path, (data_dir, model_dir, policy_dir, runs_dir))
    run_id = uuid4().hex
    directory = runs_dir / run_id
    directory.mkdir(parents=True, exist_ok=False)
    log = RunLog(directory, run_id, scenario)
    source = {}
    rows = None

    try:
        with log.stage('ingestion') as event:
            manifest_bytes = (data_dir / 'manifest.json').read_bytes()
            manifest = json.loads(manifest_bytes)
            source['data_manifest_sha256'] = _sha(manifest_bytes)
            reference = _verified_partition(data_dir, 'reference', manifest)
            batch, source['batch_sha256'] = _read_batch(data_dir, batch_path, manifest)
            rows = event['rows'] = len(batch)

        with log.stage('validation', rows) as event:
            try:
                _validate(batch)
                overlap = int(batch.ID.isin(reference.ID).sum())
                if overlap:
                    raise InvalidBatch((Violation('ID', 'reference_id_overlap', overlap),))
            except InvalidBatch as error:
                event.update(error.as_dict())
        if event['status'] == 'rejected':
            for violation in event['violations']:
                log.alert(kind='quality_violation', severity='error', rule=violation['rule'],
                          feature=violation['column'], value=violation['count'], threshold=0,
                          action='Corrigir o lote na origem e reexecutar a validação.')
            return _finish(log, 'rejected', rows, source, reason='contract_violation')

        with log.stage('transformation', rows):
            current = _validate(_transform(batch, scenario))

        with log.stage('prediction', rows) as event:
            models, model_manifest = load_models(model_dir, trusted=trusted_model)
            if model_manifest['metadata'].get('data_manifest_sha256') != source['data_manifest_sha256']:
                raise ValueError('Origem do modelo não corresponde aos dados')
            source['model_sha256'] = model_manifest['model_sha256']
            threshold = model_manifest['thresholds']['model']
            scores, performance = _predict(models.model, threshold, batch, current, scenario)
            signals = event['predictions'] = prediction_summary(scores, threshold)
            event['performance_status'] = performance['status']

        with log.stage('drift', rows) as event:
            policy = _read_policy(policy_dir, source)
            comparison = compare_drift(reference, current, policy)
            drift_status = comparison['status']
            event.update(drift_status=drift_status, alerted_features=comparison['alerted_features'])
            if drift_status != 'ok':
                event['status'] = 'unavailable'
        if drift_status != 'ok':
            required = policy['min_samples'] if drift_status == 'insufficient_samples' else policy['calibration_rows']
            log.alert(kind='drift_unavailable', severity='warning', rule=drift_status,
                      value=rows, threshold=required, status='unavailable',
                      action='Aguardar um lote de tamanho calibrado ou revisar a calibração.')
            return _finish(log, 'unavailable', rows, source, reason=drift_status)
        _emit_drift_alerts(log, comparison)

        with log.stage('report', rows) as event:
            temporary = directory / '.report-in-progress'
            temporary.mkdir()
            render_evidently(reference, current, policy, comparison, temporary / 'drift.html')
            _write_json(temporary / 'report.json', {
                'run_id': run_id, 'scenario': scenario, 'predictions': signals,
                'performance': performance, 'drift': comparison,
            })
            temporary.rename(directory / 'reports')
            event['files'] = list(REPORT_FILES)
    except Exception:
        stage = log.current_stage
        code = f'{stage}_failed'
        log.alert(kind=code, severity='error', rule=code, action=FAILURE_ACTIONS[stage])
        return _finish(log, 'failed', rows, source, reason=code)

    return _finish(log, 'succeeded', rows, source, files=REPORT_FILES)
