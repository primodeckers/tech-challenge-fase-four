"""Rastreamento local das execuções do pipeline no MLflow."""

from collections import Counter
from contextlib import suppress
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
from time import time

from mlflow import MlflowClient
from mlflow.entities import Metric, Param
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_ALREADY_EXISTS, ErrorCode

EXPERIMENT_NAME = 'credit-monitor'
STATES = {'succeeded', 'failed', 'rejected', 'unavailable'}
STAGE_METRICS = ('rows', 'duration_seconds', 'cpu_seconds', 'cpu_percent_one_core', 'rss_bytes')


def local_tracking_uri(directory):
    """URI de um arquivo SQLite no diretório local de tracking."""
    path = (Path(directory).resolve() / 'mlflow.db').as_posix()
    if path.startswith('//') or '?' in path or '#' in path:
        raise ValueError('Use um caminho local sem caracteres reservados para o tracking')
    return f'sqlite:///{path}'


def _read_run(run_dir):
    result_bytes = (run_dir / 'result.json').read_bytes()
    result = json.loads(result_bytes)
    if (result['stage'] != 'S07' or result['format_version'] != 1
            or result['run_id'] != run_dir.name or result['status'] not in STATES):
        raise ValueError('Resultado incompatível com a execução S07')
    event_bytes = (run_dir / 'events.jsonl').read_bytes()
    if ('events_sha256' in result
            and hashlib.sha256(event_bytes).hexdigest() != result['events_sha256']):
        raise ValueError('SHA-256 dos eventos divergente')
    events = [json.loads(line) for line in event_bytes.splitlines()]
    if not events or any((e['run_id'], e['scenario']) != (result['run_id'], result['scenario'])
                         for e in events):
        raise ValueError('Eventos não correspondem à execução')
    if sum(e['event'] == 'alert' for e in events) != result['alert_count']:
        raise ValueError('Contagem de alertas divergente')
    artifacts = {'result.json': result_bytes, 'events.jsonl': event_bytes}
    report = {}
    if result['status'] == 'succeeded':
        payload = (run_dir / 'reports/report.json').read_bytes()
        if hashlib.sha256(payload).hexdigest() != result['files']['reports/report.json']['sha256']:
            raise ValueError('SHA-256 do relatório agregado divergente')
        report = json.loads(payload)
        if (report['run_id'], report['scenario']) != (result['run_id'], result['scenario']):
            raise ValueError('Relatório não corresponde à execução')
        predictions = [e for e in events if e['event'] == 'stage' and e['stage'] == 'prediction']
        if (len(predictions) != 1 or predictions[0]['rows'] != result['rows']
                or predictions[0].get('predictions') != report['predictions']):
            raise ValueError('Predições divergentes entre eventos e relatório')
        artifacts['reports/report.json'] = payload
    return result, events, report, artifacts


def _numbers(prefix, values):
    for key, value in values.items():
        name = f'{prefix}.{key}'
        if isinstance(value, dict):
            yield from _numbers(name, value)
        elif type(value) in (int, float) and math.isfinite(value):
            yield name, value


def _metrics(result, events, report):
    metrics = dict(_numbers('execution', {'rows': result['rows'], 'alerts': result['alert_count']}))
    violations = Counter()
    for event in events:
        if event['event'] != 'stage':
            continue
        metrics.update(_numbers(f"execution.{event['stage']}",
                                {key: event.get(key) for key in STAGE_METRICS}))
        for violation in event.get('violations', []):
            violations[violation['rule']] += violation['count']
        metrics.update(_numbers('predictions', event.get('predictions', {})))
    metrics.update(_numbers('quality', violations))
    metrics.update(_numbers('drift', report.get('drift', {}).get('features', {})))
    metrics.update(_numbers('performance', report.get('performance', {})))
    timestamp = int(time() * 1000)
    return [Metric(key, value, timestamp, step=0) for key, value in metrics.items()]


def _experiment(client, artifact_root):
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        try:
            return client.create_experiment(EXPERIMENT_NAME, artifact_location=artifact_root)
        except MlflowException as error:
            if error.error_code != ErrorCode.Name(RESOURCE_ALREADY_EXISTS):
                raise
            experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None or experiment.artifact_location != artifact_root:
        raise ValueError('O experimento deve apontar para o diretório de artefatos local')
    return experiment.experiment_id


def track_run(run_dir, tracking_dir):
    """Persiste o resultado e retorna o ID MLflow da execução."""
    run_dir = Path(run_dir)
    result, events, report, artifacts = _read_run(run_dir)
    tracking_dir = Path(tracking_dir).resolve()
    tracking_uri = local_tracking_uri(tracking_dir)
    tracking_dir.mkdir(parents=True, exist_ok=True)
    artifact_root = (tracking_dir / 'artifacts').resolve()
    if not artifact_root.is_relative_to(tracking_dir):
        raise ValueError('Diretório de artefatos fora do armazenamento local')
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = _experiment(client, artifact_root.as_uri())
    tags = {
        'pipeline.run_id': result['run_id'], 'pipeline.scenario': result['scenario'],
        'pipeline.status': result['status'], 'tracking.state': 'recording',
        'pipeline.events_integrity': 'verified' if 'events_sha256' in result else 'legacy_unhashed',
        'pipeline.provenance': 'recorded' if 'pipeline_code_sha256' in result['source'] else 'legacy_unknown',
    }
    for name in ('reason', 'failed_stage'):
        if result[name] is not None:
            tags[f'pipeline.{name}'] = result[name]
    params = {**result['source'], 'tracking_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'tracking_python_version': platform.python_version(), 'mlflow_version': version('mlflow')}
    prediction = next((e.get('predictions') for e in events if e['stage'] == 'prediction'
                       and e.get('predictions') is not None), None)
    if prediction is not None:
        params['decision_threshold'] = prediction['threshold']
    run = client.create_run(experiment_id, tags=tags,
                            run_name=f"{result['scenario']} {result['run_id'][:8]}")
    run_id = run.info.run_id
    try:
        client.log_batch(run_id, metrics=_metrics(result, events, report),
                         params=[Param(key, str(value)) for key, value in params.items()],
                         synchronous=True)
        with TemporaryDirectory() as temporary:
            for name, payload in artifacts.items():
                subdir, _, filename = name.rpartition('/')
                copy = Path(temporary) / filename
                copy.write_bytes(payload)
                client.log_artifact(run_id, str(copy), artifact_path=subdir or None)
        client.set_tag(run_id, 'tracking.state', 'complete')
        client.set_terminated(run_id, status='FINISHED' if result['status'] == 'succeeded' else 'FAILED')
    except Exception:
        with suppress(Exception):
            client.set_terminated(run_id, status='FAILED')
        with suppress(Exception):
            client.set_tag(run_id, 'tracking.state', 'failed')
        raise
    return run_id
