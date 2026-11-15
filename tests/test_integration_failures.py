"""Estados de erro reais da CLI e preservação de execuções anteriores."""

import json

from mlflow import MlflowClient
import pandas as pd
import pytest

from credit_monitor.tracking import local_tracking_uri
from test_integration import invoke


@pytest.mark.parametrize('case,code,status', [
    ('empty', 2, 'rejected'), ('extra_column', 2, 'rejected'), ('small', 3, 'unavailable'),
])
def test_invalid_or_unavailable_batches_keep_their_state_through_tracking(inputs, tmp_path,
                                                                        capsys, case, code, status):
    frame = pd.read_csv(inputs['data_dir'] / 'production.csv')
    if case == 'empty':
        frame = frame.iloc[:0]
    elif case == 'extra_column':
        frame['private_field'] = 'PRIVATE-CONTENT'
    else:
        frame = frame.iloc[:32]
    batch = tmp_path / 'batch.csv'
    frame.to_csv(batch, index=False)
    tracking = tmp_path / 'tracking'
    actual_code, output = invoke(inputs, tracking, capsys, extra=['--batch', str(batch)])
    assert actual_code == code and output['status'] == status
    run_dir = inputs['runs_dir'] / output['run_id']
    result = json.loads((run_dir / 'result.json').read_text())
    assert result['files'] == {} and not (run_dir / 'reports').exists()
    client = MlflowClient(tracking_uri=local_tracking_uri(tracking))
    run = client.get_run(output['tracking']['mlflow_run_id'])
    assert run.info.status == 'FAILED' and run.data.tags['pipeline.status'] == status
    assert {item.path for item in client.list_artifacts(run.info.run_id)} == {'events.jsonl', 'result.json'}
    events = (run_dir / 'events.jsonl').read_text()
    assert 'PRIVATE-CONTENT' not in events and 'private_field' not in events
    if status == 'rejected':
        assert not any(key.startswith('predictions.') for key in run.data.metrics)


def test_real_tracking_failure_does_not_change_success_or_previous_outputs(inputs, tmp_path, capsys):
    tracking = tmp_path / 'tracking'
    code, first = invoke(inputs, tracking, capsys)
    assert code == 0
    first_dir = inputs['runs_dir'] / first['run_id']
    originals = {path.relative_to(first_dir): path.read_bytes()
                 for path in first_dir.rglob('*') if path.is_file()}
    blocked = tmp_path / 'tracking-is-a-file'
    blocked.write_text('keep')
    code, second = invoke(inputs, blocked, capsys)
    assert code == 4 and second['status'] == 'succeeded'
    assert second['tracking']['status'] == 'failed'
    assert second['tracking']['reason'] == 'tracking_failed'
    assert first['run_id'] != second['run_id']
    assert blocked.read_text() == 'keep'
    assert all((first_dir / name).read_bytes() == payload for name, payload in originals.items())
    second_dir = inputs['runs_dir'] / second['run_id']
    assert json.loads((second_dir / 'result.json').read_text())['status'] == 'succeeded'
    assert json.loads((second_dir / 'tracking.json').read_text()) == second['tracking']


def test_real_storage_failure_returns_failure_without_fabricating_a_run(inputs, tmp_path, capsys):
    blocked = tmp_path / 'runs-is-a-file'
    blocked.write_text('keep')
    inputs['runs_dir'] = blocked
    code, output = invoke(inputs, tmp_path / 'tracking', capsys)
    assert code == 1 and output['status'] == 'failed'
    assert output['reason'] == 'run_storage_failed' and output['run_id'] is None
    assert blocked.read_text() == 'keep'
    assert not (tmp_path / 'tracking').exists()
