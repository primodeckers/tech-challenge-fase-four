import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_monitor import pipeline
from credit_monitor.baseline import train_models
from credit_monitor.data import TARGET
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
    return dict(data_dir=data, model_dir=model, runs_dir=runs, trusted_model=True)


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
