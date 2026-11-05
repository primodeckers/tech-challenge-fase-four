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
