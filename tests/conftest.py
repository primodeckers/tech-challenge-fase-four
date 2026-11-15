"""Fixtures sintéticas compartilhadas. A suíte não acessa a rede."""

import hashlib
import json
import socket

import numpy as np
import pandas as pd
import pytest

from credit_monitor.baseline import train_models
from credit_monitor.data import TARGET
from credit_monitor.drift import calibrate_policy
from credit_monitor.model_store import save_models


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("A suíte deve usar dados sintéticos sem rede")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)


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

