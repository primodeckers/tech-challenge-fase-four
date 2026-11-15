"""Contratos entre CLI, pipeline e SQLite real, usando apenas dados sintéticos."""

import json
from pathlib import Path

from mlflow import MlflowClient
import pandas as pd
import pytest

from credit_monitor import cli
from credit_monitor.data import TARGET
from credit_monitor.tracking import local_tracking_uri


def invoke(inputs, tracking, capsys, *, scenario='control', extra=()):
    arguments = [scenario, '--trusted-model', '--tracking-dir', str(tracking)]
    for name in ('data_dir', 'model_dir', 'policy_dir', 'runs_dir'):
        arguments.extend(['--' + name.replace('_', '-'), str(inputs[name])])
    code = cli.main([*arguments, *extra])
    return code, json.loads(capsys.readouterr().out)


@pytest.mark.parametrize('scenario,alerts', [
    ('control', []), ('data_drift', ['LIMIT_BAL', 'PAY_AMT1']), ('concept_drift', []),
])
def test_cli_records_real_scenarios_and_matching_alerts_in_mlflow(inputs, tmp_path, capsys,
                                                                scenario, alerts):
    tracking = tmp_path / 'tracking'
    code, output = invoke(inputs, tracking, capsys, scenario=scenario)
    assert code == 0 and output['status'] == 'succeeded'
    run_dir = inputs['runs_dir'] / output['run_id']
    report = json.loads((run_dir / 'reports/report.json').read_text())
    assert report['drift']['alerted_features'] == alerts
    assert output['alert_count'] == len(alerts)
    client = MlflowClient(tracking_uri=local_tracking_uri(tracking))
    run = client.get_run(output['tracking']['mlflow_run_id'])
    assert run.info.status == 'FINISHED'
    assert run.data.tags['pipeline.run_id'] == output['run_id']
    assert run.data.tags['pipeline.events_integrity'] == 'verified'
    assert run.data.tags['pipeline.provenance'] == 'recorded'
    assert run.data.metrics['execution.rows'] == 240
    assert run.data.metrics['predictions.positive_rate'] == report['predictions']['positive_rate']
    downloaded = client.download_artifacts(run.info.run_id, 'events.jsonl', str(tmp_path))
    assert Path(downloaded).read_bytes() == (run_dir / 'events.jsonl').read_bytes()
    assert json.loads((run_dir / 'tracking.json').read_text()) == output['tracking']


def test_unlabeled_cli_keeps_predictions_without_inventing_performance_in_tracking(inputs, tmp_path,
                                                                                capsys):
    batch = tmp_path / 'unlabeled.csv'
    pd.read_csv(inputs['data_dir'] / 'production.csv').drop(columns=TARGET).to_csv(batch, index=False)
    tracking = tmp_path / 'tracking'
    code, output = invoke(inputs, tracking, capsys, extra=['--batch', str(batch)])
    assert code == 0
    report = json.loads((inputs['runs_dir'] / output['run_id'] / 'reports/report.json').read_text())
    assert report['performance'] == {'status': 'unavailable_without_labels'}
    run = MlflowClient(tracking_uri=local_tracking_uri(tracking)).get_run(
        output['tracking']['mlflow_run_id'])
    assert 'predictions.positive_rate' in run.data.metrics
    assert not any(key.startswith('performance.') for key in run.data.metrics)

