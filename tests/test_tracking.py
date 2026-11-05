import hashlib
import json

from mlflow import MlflowClient
import pytest

from credit_monitor.tracking import local_tracking_uri, track_run


def make_run(root, status='succeeded'):
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
    (run_dir / 'events.jsonl').write_text(''.join(json.dumps(event) + '\n' for event in events))
    result = {**shared, 'stage': 'S07', 'format_version': 1, 'status': status,
              'rows': 240, 'alert_count': int(status == 'rejected'), 'files': {},
              'failed_stage': 'validation' if status == 'rejected' else None,
              'reason': 'contract_violation' if status == 'rejected' else None,
              'source': {'data_manifest_sha256': 'a' * 64, 'drift_seed': 42}}
    if status == 'succeeded':
        report = run_dir / 'reports/report.json'
        report.parent.mkdir()
        report.write_text(json.dumps({**shared, 'performance': {'metrics': {'f1': .4}},
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
    assert len(a.data.params['code_sha256']) == 64
    assert {item.path for item in client.list_artifacts(a.info.run_id)} == {
        'result.json', 'events.jsonl', 'reports'}
    assert [item.path for item in client.list_artifacts(a.info.run_id, 'reports')] == ['reports/report.json']
    assert {item.path for item in client.list_artifacts(b.info.run_id)} == {'result.json', 'events.jsonl'}


def test_tracker_rejects_report_that_does_not_match_terminal_hash(tmp_path):
    directory = make_run(tmp_path)
    (directory / 'reports/report.json').write_text('{}')
    with pytest.raises(ValueError, match='SHA-256'):
        track_run(directory, tmp_path / 'tracking')


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
