import hashlib
import json

from mlflow import MlflowClient
import pytest

from credit_monitor.tracking import local_tracking_uri, track_run


def make_run(root, status='succeeded', *, legacy=False):
    run_dir = root / ('1' * 32 if status == 'succeeded' else '2' * 32)
    run_dir.mkdir(parents=True)
    run_id = run_dir.name
    shared = {'run_id': run_id, 'scenario': 'control'}
    events = [{**shared, 'event': 'stage', 'stage': 'validation', 'rows': 240,
               'duration_seconds': .12, 'cpu_seconds': .1, 'cpu_percent_one_core': 83,
               'rss_bytes': 4000}]
    if status == 'rejected':
        events[0]['violations'] = [{'rule': 'adult_age_range', 'count': 2}]
        events.append({**shared, 'event': 'alert', 'stage': 'validation', 'type': 'quality_violation',
                       'severity': 'error', 'rule': 'adult_age_range', 'value': 2, 'threshold': 0})
    elif status == 'succeeded':
        events.append({**shared, 'event': 'stage', 'stage': 'prediction', 'rows': 240,
                       'predictions': {'threshold': .27, 'positive_count': 60, 'positive_rate': .25}})
    event_bytes = ''.join(json.dumps(event) + '\n' for event in events).encode()
    (run_dir / 'events.jsonl').write_bytes(event_bytes)
    result = {**shared, 'stage': 'S07', 'format_version': 1, 'status': status,
              'rows': 240, 'alert_count': int(status == 'rejected'), 'files': {},
              'failed_stage': 'validation' if status == 'rejected' else None,
              'reason': 'contract_violation' if status == 'rejected' else None,
              'source': {'data_manifest_sha256': 'a' * 64, 'drift_seed': 42}}
    if not legacy:
        result['events_sha256'] = hashlib.sha256(event_bytes).hexdigest()
        result['source'].update(pipeline_code_sha256='d' * 64, pipeline_python_version='3.12.9')
    if status == 'succeeded':
        report = run_dir / 'reports/report.json'
        report.parent.mkdir()
        report.write_text(json.dumps({**shared, 'performance': {'metrics': {'f1': .4}},
                                      'predictions': events[1]['predictions'],
                                      'drift': {'features': {'AGE': {'psi': .12, 'ks': {'statistic': .14}}}}}))
        result['files']['reports/report.json'] = {'sha256': hashlib.sha256(report.read_bytes()).hexdigest()}
    (run_dir / 'result.json').write_text(json.dumps(result))
    return run_dir


def test_sqlite_retains_metrics_artifacts_and_distinguishes_rejection(tmp_path):
    tracking_dir = tmp_path / 'tracking with spaces'
    client = MlflowClient(tracking_uri=local_tracking_uri(tracking_dir))
    success = make_run(tmp_path)
    rejected = make_run(tmp_path, 'rejected')
    a = client.get_run(track_run(success, tracking_dir))
    b = client.get_run(track_run(rejected, tracking_dir))

    assert a.info.status == 'FINISHED' and b.info.status == 'FAILED'
    assert a.data.tags['pipeline.run_id'] == success.name
    assert b.data.tags['pipeline.status'] == 'rejected'
    assert b.data.tags['pipeline.reason'] == 'contract_violation'
    assert a.data.metrics['execution.validation.rss_bytes'] == 4000
    assert a.data.metrics['predictions.positive_rate'] == .25
    assert a.data.metrics['drift.AGE.psi'] == .12
    assert a.data.metrics['performance.metrics.f1'] == .4
    assert b.data.metrics['quality.adult_age_range'] == 2
    assert 'predictions.positive_rate' not in b.data.metrics
    assert a.data.params['drift_seed'] == '42'
    assert a.data.params['pipeline_code_sha256'] == 'd' * 64
    assert a.data.params['pipeline_python_version'] == '3.12.9'
    assert len(a.data.params['tracking_code_sha256']) == 64
    assert a.data.tags['pipeline.events_integrity'] == 'verified'
    assert {item.path for item in client.list_artifacts(a.info.run_id)} == {
        'result.json', 'events.jsonl', 'reports'}
    assert [item.path for item in client.list_artifacts(a.info.run_id, 'reports')] == ['reports/report.json']
    assert {item.path for item in client.list_artifacts(b.info.run_id)} == {'result.json', 'events.jsonl'}


def test_tracker_rejects_report_that_does_not_match_terminal_hash(tmp_path):
    directory = make_run(tmp_path)
    (directory / 'reports/report.json').write_text('{}')
    with pytest.raises(ValueError, match='SHA-256'):
        track_run(directory, tmp_path / 'tracking')


@pytest.mark.parametrize('legacy,reason', [(False, 'SHA-256 dos eventos'), (True, 'Predições divergentes')])
def test_tracker_rejects_events_that_disagree_with_the_result(tmp_path, legacy, reason):
    directory = make_run(tmp_path, legacy=legacy)
    events = directory / 'events.jsonl'
    events.write_bytes(events.read_bytes().replace(b'"positive_rate": 0.25', b'"positive_rate": 0.90'))
    with pytest.raises(ValueError, match=reason):
        track_run(directory, tmp_path / 'tracking')


def test_legacy_tracking_does_not_invent_pipeline_provenance(tmp_path):
    directory = make_run(tmp_path, legacy=True)
    tracking_dir = tmp_path / 'tracking'
    run_id = track_run(directory, tracking_dir)
    run = MlflowClient(tracking_uri=local_tracking_uri(tracking_dir)).get_run(run_id)
    assert run.info.status == 'FINISHED'
    assert 'code_sha256' not in run.data.params
    assert 'pipeline_code_sha256' not in run.data.params
    assert run.data.tags['pipeline.events_integrity'] == 'legacy_unhashed'
    assert run.data.tags['pipeline.provenance'] == 'legacy_unknown'


def test_tracking_error_marks_partial_mlflow_run_failed(tmp_path, monkeypatch):
    directory = make_run(tmp_path)
    tracking_dir = tmp_path / 'tracking'

    def fail(*args, **kwargs):
        raise OSError('artifact storage failed')

    monkeypatch.setattr(MlflowClient, 'log_artifact', fail)
    with pytest.raises(OSError):
        track_run(directory, tracking_dir)
    client = MlflowClient(tracking_uri=local_tracking_uri(tracking_dir))
    experiment = client.get_experiment_by_name('credit-monitor')
    run, = client.search_runs([experiment.experiment_id])
    assert run.info.status == 'FAILED'
    assert run.data.tags['tracking.state'] == 'failed'
