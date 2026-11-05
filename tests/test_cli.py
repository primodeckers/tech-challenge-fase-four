import json

import pytest

from credit_monitor import cli


@pytest.mark.parametrize('status,expected', [
    ('succeeded', 0), ('failed', 1), ('rejected', 2), ('unavailable', 3),
])
def test_cli_outputs_a_terminal_summary_and_distinct_exit_code(monkeypatch, capsys, status, expected):
    def fake_run(**arguments):
        assert arguments['scenario'] == 'data_drift'
        assert arguments['batch_path'].name == 'test.csv'
        assert arguments['trusted_model'] is True
        return {'run_id': 'run-123', 'scenario': 'data_drift', 'status': status,
                'rows': 25, 'failed_stage': None, 'reason': None, 'alert_count': 0}

    monkeypatch.setattr(cli, 'run_pipeline', fake_run)
    code = cli.main(['data_drift', '--batch', 'test.csv', '--trusted-model'])
    payload = json.loads(capsys.readouterr().out)
    assert code == expected and payload['status'] == status and payload['run_id'] == 'run-123'


def test_cli_reports_storage_failure_without_paths_or_raw_errors(monkeypatch, capsys):
    def fail(**arguments):
        raise PermissionError('SENSITIVE-PATH')

    monkeypatch.setattr(cli, 'run_pipeline', fail)
    assert cli.main(['control']) == 1
    output = capsys.readouterr()
    assert json.loads(output.out)['reason'] == 'run_storage_failed'
    assert 'SENSITIVE-PATH' not in output.out + output.err


@pytest.mark.parametrize('pipeline_state,tracking_ok,expected', [
    ('succeeded', True, 0), ('succeeded', False, 4), ('rejected', False, 2),
])
def test_requested_tracking_has_a_receipt_and_explicit_failure(tmp_path, monkeypatch, capsys,
                                                               pipeline_state, tracking_ok, expected):
    from credit_monitor import tracking

    runs = tmp_path / 'runs'
    run_id = 'a' * 32
    directory = runs / run_id
    directory.mkdir(parents=True)
    result = {'run_id': run_id, 'scenario': 'control', 'status': pipeline_state,
              'rows': 25, 'failed_stage': None, 'reason': None, 'alert_count': 0}
    original = json.dumps(result)
    (directory / 'result.json').write_text(original)
    monkeypatch.setattr(cli, 'run_pipeline', lambda **kwargs: result)

    def record(run_dir, tracking_dir):
        assert run_dir == directory and tracking_dir == tmp_path / 'tracking'
        if not tracking_ok:
            raise OSError('SENSITIVE-TRACKING-DETAIL')
        return 'mlflow-id'

    monkeypatch.setattr(tracking, 'track_run', record)
    code = cli.main(['control', '--runs-dir', str(runs), '--tracking-dir', str(tmp_path / 'tracking')])
    raw = capsys.readouterr().out
    output = json.loads(raw)
    receipt = json.loads((directory / 'tracking.json').read_text())
    assert code == expected and output['status'] == pipeline_state
    assert output['tracking'] == receipt
    assert receipt['status'] == ('recorded' if tracking_ok else 'failed')
    assert receipt['mlflow_run_id'] == ('mlflow-id' if tracking_ok else None)
    assert 'SENSITIVE-TRACKING-DETAIL' not in raw
    assert (directory / 'result.json').read_text() == original
