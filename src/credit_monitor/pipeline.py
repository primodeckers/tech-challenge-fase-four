"""Execuções independentes dos cenários, com contrato e inferência congelada."""

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
}


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _read_batch(data_dir, batch_path, manifest):
    if batch_path is None:
        partition = manifest['partitions']['production']
        if partition['file'] != 'production.csv':
            raise ValueError('Nome de partição inesperado')
        batch_path = data_dir / 'production.csv'
    payload = Path(batch_path).read_bytes()
    digest = _sha(payload)
    if batch_path == data_dir / 'production.csv' and digest != manifest['partitions']['production']['sha256']:
        raise ValueError('SHA-256 de produção divergente')
    frame = pd.read_csv(BytesIO(payload))
    if not isinstance(frame.index, pd.RangeIndex):
        raise ValueError('CSV com campos adicionais fora do cabeçalho')
    if batch_path == data_dir / 'production.csv' and len(frame) != manifest['partitions']['production']['rows']:
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


def run_pipeline(*, data_dir, model_dir, runs_dir, scenario='control', batch_path=None,
                 trusted_model=False):
    """Retorna estado terminal e mantém o histórico local em uma pasta nova."""
    if scenario not in SCENARIOS:
        raise ValueError('Cenário desconhecido')
    data_dir, model_dir, runs_dir = map(Path, (data_dir, model_dir, runs_dir))
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
            event['predictions'] = prediction_summary(scores, threshold)
            event['performance_status'] = performance['status']
    except Exception:
        stage = log.current_stage
        code = f'{stage}_failed'
        log.alert(kind=code, severity='error', rule=code, action=FAILURE_ACTIONS[stage])
        return _finish(log, 'failed', rows, source, reason=code)

    return _finish(log, 'succeeded', rows, source)
