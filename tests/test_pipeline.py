import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_monitor import pipeline
from credit_monitor.baseline import train_models
from credit_monitor.data import TARGET
from credit_monitor.drift import calibrate_policy
from credit_monitor.model_store import save_models


@pytest.fixture
def inputs(tmp_path):
    rows = 240
    steps = np.arange(rows)
    values = {'ID': steps, 'LIMIT_BAL': 20000 + steps * 1000,
              'SEX': np.tile([1, 2], rows // 2), 'EDUCATION': np.ones(rows),
              'MARRIAGE': np.ones(rows), 'AGE': 25 + steps % 50,
              TARGET: (steps % 3 == 0).astype(int)}
    for column in ('PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'):
        values[column] = steps % 4 - 1
    for i in range(1, 7):
        values[f'BILL_AMT{i}'] = steps * 100
        values[f'PAY_AMT{i}'] = steps * 10
    reference = pd.DataFrame(values, dtype='int64')
    data, model, runs = (tmp_path / name for name in ('data', 'model', 'runs'))
    data.mkdir()
    source = {'partitions': {}}
    for offset, name in enumerate(('reference', 'calibration', 'production')):
        frame = reference.assign(ID=reference.ID + 1000 * offset)
        path = data / f'{name}.csv'
        frame.to_csv(path, index=False)
        source['partitions'][name] = {'file': path.name, 'rows': len(frame),
                                     'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    payload = json.dumps(source).encode()
    (data / 'manifest.json').write_bytes(payload)
    save_models(train_models(reference), model, thresholds={'model': .27, 'dummy': .5},
                metadata={'data_manifest_sha256': hashlib.sha256(payload).hexdigest()})
    policy_dir = tmp_path / 'policy'
    policy_dir.mkdir()
    policy = calibrate_policy(reference, reference.assign(ID=reference.ID + 1000), n_resamples=20)
    policy_bytes = json.dumps(policy).encode()
    (policy_dir / 'policy.json').write_bytes(policy_bytes)
    (policy_dir / 'manifest.json').write_text(json.dumps({
        'stage': 'S05', 'format_version': 1,
        'data_manifest_sha256': hashlib.sha256(payload).hexdigest(),
        'files': {'policy.json': {'sha256': hashlib.sha256(policy_bytes).hexdigest()}},
    }))
    return dict(data_dir=data, model_dir=model, policy_dir=policy_dir, runs_dir=runs,
                trusted_model=True)


def _events(inputs, result):
    path = inputs['runs_dir'] / result['run_id'] / 'events.jsonl'
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def test_invalid_batch_is_logged_without_loading_model_or_leaking_input(inputs, monkeypatch, tmp_path):
    batch = pd.read_csv(inputs['data_dir'] / 'production.csv')
    batch.loc[0, 'AGE'] = 11
    batch['secret_customer_field'] = 'SENSITIVE-INPUT'
    path = tmp_path / 'invalid.csv'
    batch.to_csv(path, index=False)
    monkeypatch.setattr(pipeline, 'load_models', lambda *a, **k: pytest.fail('modelo carregado'))
    result = pipeline.run_pipeline(**inputs, batch_path=path)
    assert result['status'] == 'rejected' and result['failed_stage'] == 'validation'
    events = _events(inputs, result)
    assert [e['stage'] for e in events if e['event'] == 'stage'] == ['ingestion', 'validation']
    assert any(e['rule'] == 'adult_age_range' and e['value'] == 1
               for e in events if e['event'] == 'alert')
    assert 'secret_customer_field' not in json.dumps(events)
    assert 'SENSITIVE-INPUT' not in json.dumps(events)


@pytest.mark.parametrize('failure', ['ingestion', 'transformation', 'prediction'])
def test_stage_failures_are_distinct_and_sanitized(inputs, monkeypatch, tmp_path, failure):
    def fail(*args, **kwargs):
        raise ValueError('SENSITIVE-EXCEPTION')

    arguments = {}
    if failure == 'ingestion':
        arguments['batch_path'] = tmp_path / 'missing.csv'
    elif failure == 'transformation':
        monkeypatch.setattr(pipeline, 'build_scenarios', fail)
        arguments['scenario'] = 'data_drift'
    else:
        monkeypatch.setattr(pipeline, 'predict_scores', fail)
    result = pipeline.run_pipeline(**inputs, **arguments)
    assert result['status'] == 'failed' and result['failed_stage'] == failure
    assert result['files'] == {}
    events = _events(inputs, result)
    assert events[-2]['stage'] == events[-1]['stage'] == failure
    assert events[-2]['status'] == 'failed' and events[-1]['type'] == f'{failure}_failed'
    assert 'SENSITIVE-EXCEPTION' not in json.dumps(events)


def test_unlabeled_batch_keeps_model_signals_without_performance(inputs, tmp_path):
    frame = pd.read_csv(inputs['data_dir'] / 'production.csv').drop(columns=TARGET)
    path = tmp_path / 'unlabeled.csv'
    frame.to_csv(path, index=False)
    result = pipeline.run_pipeline(**inputs, batch_path=path)
    assert result['status'] == 'succeeded'
    prediction = next(e for e in _events(inputs, result) if e['stage'] == 'prediction')
    assert prediction['performance_status'] == 'unavailable_without_labels'
    assert sum(prediction['predictions']['probability_histogram']['counts']) == len(frame)
    assert prediction['predictions']['threshold'] == .27


def test_ingestion_rejects_csv_with_implicit_index(inputs, tmp_path):
    source = inputs['data_dir'] / 'production.csv'
    header, *lines = source.read_text().splitlines()
    path = tmp_path / 'extra-field.csv'
    path.write_text(header + '\n' + '\n'.join('SECRET,' + line for line in lines))
    result = pipeline.run_pipeline(**inputs, batch_path=path)
    assert result['status'] == 'failed' and result['failed_stage'] == 'ingestion'
    assert 'SECRET' not in json.dumps(_events(inputs, result))


def test_drift_alert_report_and_manifest_agree_on_one_comparison(inputs, monkeypatch):
    compare = pipeline.compare_drift
    comparisons = []

    def record(*args):
        result = compare(*args)
        comparisons.append(result)
        return result

    monkeypatch.setattr(pipeline, 'compare_drift', record)
    result = pipeline.run_pipeline(**inputs, scenario='data_drift')
    run_dir = inputs['runs_dir'] / result['run_id']
    assert result['status'] == 'succeeded' and len(comparisons) == 1
    assert result == json.loads((run_dir / 'result.json').read_text())
    report = json.loads((run_dir / 'reports/report.json').read_text())
    assert report['run_id'] == result['run_id'] and report['drift'] == comparisons[0]
    assert report['performance']['status'] == 'unavailable_under_feature_intervention'
    events = _events(inputs, result)
    assert all(e['run_id'] == result['run_id'] for e in events)
    alerts = [e for e in events if e['event'] == 'alert']
    assert {e['feature'] for e in alerts} == {'LIMIT_BAL', 'PAY_AMT1'}
    assert all(e['type'] == 'data_drift' and e['value']['ks_d'] > e['threshold']['ks_d']
               for e in alerts)
    assert result['alert_count'] == len(alerts)
    assert (run_dir / 'reports/drift.html').stat().st_size > 1000
    for name, info in result['files'].items():
        assert hashlib.sha256((run_dir / name).read_bytes()).hexdigest() == info['sha256']


def test_concept_reports_changed_labels_without_claiming_feature_drift(inputs):
    result = pipeline.run_pipeline(**inputs, scenario='concept_drift')
    assert result['status'] == 'succeeded'
    path = inputs['runs_dir'] / result['run_id'] / 'reports/report.json'
    report = json.loads(path.read_text())
    assert report['performance']['changed_labels']['total'] > 0
    assert report['performance']['label_marginal_unchanged']
    assert report['drift']['alerted_features'] == []


@pytest.mark.parametrize('rows,reason,threshold', [
    (12, 'insufficient_samples', 200), (230, 'uncalibrated_sample_size', 240),
])
def test_unavailable_drift_has_an_alert_and_no_success_report(inputs, tmp_path, rows, reason, threshold):
    batch = pd.read_csv(inputs['data_dir'] / 'production.csv').iloc[:rows]
    path = tmp_path / 'small.csv'
    batch.to_csv(path, index=False)
    result = pipeline.run_pipeline(**inputs, batch_path=path)
    assert result['status'] == 'unavailable' and result['reason'] == reason
    assert result['failed_stage'] == 'drift' and result['files'] == {}
    events = _events(inputs, result)
    assert events[-1]['value'] == rows and events[-1]['threshold'] == threshold
    assert events[-1]['status'] == 'unavailable'
    assert 'report' not in [e['stage'] for e in events]


def test_tampered_policy_stops_before_report(inputs):
    policy = inputs['policy_dir'] / 'policy.json'
    policy.write_bytes(policy.read_bytes() + b' ')
    result = pipeline.run_pipeline(**inputs)
    assert result['status'] == 'failed' and result['failed_stage'] == 'drift'
    assert not (inputs['runs_dir'] / result['run_id'] / 'reports').exists()


def test_report_failure_does_not_reuse_previous_report(inputs, monkeypatch):
    first = pipeline.run_pipeline(**inputs)
    first_report = inputs['runs_dir'] / first['run_id'] / 'reports/report.json'
    original = first_report.read_bytes()

    def fail(reference, current, policy, comparison, output):
        output.write_text('partial')
        raise OSError('SENSITIVE-REPORT-DETAIL')

    monkeypatch.setattr(pipeline, 'render_evidently', fail)
    second = pipeline.run_pipeline(**inputs)
    assert second['run_id'] != first['run_id']
    assert second['status'] == 'failed' and second['failed_stage'] == 'report'
    assert second['files'] == {} and first_report.read_bytes() == original
    second_dir = inputs['runs_dir'] / second['run_id']
    assert not (second_dir / 'reports').exists()
    events = _events(inputs, second)
    assert events[-1]['type'] == 'report_failed'
    assert 'SENSITIVE-REPORT-DETAIL' not in json.dumps(events)
