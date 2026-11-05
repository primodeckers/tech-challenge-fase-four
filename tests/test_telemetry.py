import json

import pytest

from credit_monitor import telemetry


def test_stage_records_elapsed_cpu_and_resident_memory(tmp_path, monkeypatch):
    wall = iter([10.0, 12.0])
    cpu = iter([5.0, 8.0])
    monkeypatch.setattr(telemetry, 'perf_counter', lambda: next(wall))
    monkeypatch.setattr(telemetry, 'process_time', lambda: next(cpu))
    monkeypatch.setattr(telemetry, 'resident_bytes', lambda: 4096)
    log = telemetry.RunLog(tmp_path, 'run-1', 'control')
    with log.stage('validation', rows=23):
        pass
    event = json.loads((tmp_path / 'events.jsonl').read_text())
    assert event['run_id'] == 'run-1'
    assert event['scenario'] == 'control'
    assert event['stage'] == 'validation'
    assert event['status'] == 'ok' and event['rows'] == 23
    assert event['duration_seconds'] == 2
    assert event['cpu_seconds'] == 3
    assert event['cpu_percent_one_core'] == 150
    assert event['rss_bytes'] == 4096


def test_stage_failure_and_unavailable_measurements_are_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr(telemetry, 'perf_counter', lambda: 10.0)
    monkeypatch.setattr(telemetry, 'resident_bytes', lambda: None)
    log = telemetry.RunLog(tmp_path, 'run-2', 'control')
    with pytest.raises(ValueError):
        with log.stage('prediction'):
            raise ValueError('sensitive injected value')
    raw = (tmp_path / 'events.jsonl').read_text()
    event = json.loads(raw)
    assert 'sensitive injected value' not in raw
    assert event['status'] == 'failed'
    assert event['rss_bytes'] is None and event['cpu_percent_one_core'] is None


def test_probability_histogram_includes_one_and_counts_threshold_equality():
    result = telemetry.prediction_summary([0.0, 0.05, 0.27, 0.9, 1.0], 0.27)
    assert result['positive_count'] == 3 and result['positive_rate'] == .6
    assert result['probability_histogram']['counts'] == [2, 0, 1, 0, 0, 0, 0, 0, 0, 2]
    assert result['probability_histogram']['edges'] == pytest.approx([i / 10 for i in range(11)])
