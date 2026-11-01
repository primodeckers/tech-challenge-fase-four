import numpy as np
import pandas as pd
import pytest
import hashlib
import json
from pathlib import Path

from credit_monitor.contracts import InvalidBatch
from credit_monitor.data import TARGET
from credit_monitor.scenarios import build_concept_scenario
from credit_monitor.concept import evaluate_concept, predict_concept
from credit_monitor.baseline import predict_scores, train_models
from credit_monitor.model_store import save_models
from credit_monitor.concept_run import run_concept


@pytest.fixture
def production():
    n = 32
    values = {"ID": np.arange(1001, 1001+n), "LIMIT_BAL": np.full(n, 100000),
              "SEX": np.tile([1, 2], n//2), "EDUCATION": np.ones(n),
              "MARRIAGE": np.ones(n), "AGE": np.full(n, 35)}
    for col in ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"):
        values[col] = np.repeat([-1, 2], n//2)
    for i in range(1, 7):
        values[f"BILL_AMT{i}"] = np.full(n, 1000)
        values[f"PAY_AMT{i}"] = np.full(n, 100)
    values[TARGET] = np.repeat([0, 1], n//2)
    return pd.DataFrame(values, dtype="int64")


def test_concept_preserves_x_source_and_prevalence_with_exact_conditional_swaps(production):
    original = production.copy(deep=True)
    changed = build_concept_scenario(production)
    pd.testing.assert_frame_equal(changed.drop(columns=TARGET), original.drop(columns=TARGET))
    pd.testing.assert_frame_equal(production, original)
    up = (original[TARGET] == 0) & (changed[TARGET] == 1)
    down = (original[TARGET] == 1) & (changed[TARGET] == 0)
    assert up.sum() == down.sum() == 4
    assert (production.loc[up, 'PAY_0'] <= 0).all()
    assert (production.loc[down, 'PAY_0'] > 0).all()
    assert changed[TARGET].sum() == original[TARGET].sum()
    assert all(dtype == np.dtype('int64') for dtype in changed.dtypes)
    changed.loc[0, 'AGE'] = 50
    pd.testing.assert_frame_equal(production, original)


def test_concept_seed_is_repeatable_and_id_selection_independent_of_row_order(production):
    first = build_concept_scenario(production, seed=42)
    second = build_concept_scenario(production, seed=42)
    pd.testing.assert_frame_equal(first, second)
    shuffled = production.sample(frac=1, random_state=9)
    actual = build_concept_scenario(shuffled, seed=42)
    pd.testing.assert_frame_equal(actual.sort_values('ID'), first.sort_values('ID'))
    assert not first[TARGET].equals(build_concept_scenario(production, seed=17)[TARGET])


@pytest.mark.parametrize('case', ['no_labels', 'no_delay_group', 'no_eligible_labels', 'too_small'])
def test_concept_requires_labels_and_effective_intervention(production, case):
    if case == 'no_labels':
        production = production.drop(columns=TARGET)
    elif case == 'no_delay_group':
        production['PAY_0'] = 0
    elif case == 'no_eligible_labels':
        production[TARGET] = 0
    else:
        production = production.iloc[[0, 1, 16, 17]]
    with pytest.raises(ValueError):
        build_concept_scenario(production)


@pytest.mark.parametrize('seed', [-1, True, 1.2, '42', None])
def test_concept_rejects_invalid_seed(production, seed):
    with pytest.raises(ValueError):
        build_concept_scenario(production, seed=seed)


def test_concept_validates_input_contract(production):
    production.loc[0, 'AGE'] = 18
    with pytest.raises(InvalidBatch):
        build_concept_scenario(production)


def test_evaluation_has_hand_calculated_global_and_conditional_errors(production):
    # Controle perfeito; 4 trocas em cada grupo com scores fixos.
    scores = np.repeat([0.1, 0.9], 16)
    result = evaluate_concept(production, build_concept_scenario(production), scores, 0.5)
    assert result['status'] == 'available'
    a, b = result['control'], result['concept_drift']
    assert a['confusion_matrix'] == [[16, 0], [0, 16]]
    assert b['confusion_matrix'] == [[12, 4], [4, 12]]
    assert b['roc_auc'] == pytest.approx(0.75)
    assert b['f1'] == pytest.approx(0.75)
    assert b['brier'] == pytest.approx(0.21)
    assert result['delta']['prevalence'] == 0
    assert result['delta']['brier'] == pytest.approx(0.20)
    assert result['changed_labels'] == {'total': 8, 'zero_to_one': 4, 'one_to_zero': 4}
    assert result['label_marginal_unchanged'] is True
    low, high = result['segments']['pay_0_le_0'], result['segments']['pay_0_gt_0']
    assert low['rows'] == high['rows'] == 16
    assert low['control']['false_negative_rate'] is None  # sem positivos reais
    assert low['concept_drift']['false_negative_rate'] == 1
    assert low['concept_drift']['actual_positives'] == 4
    assert high['concept_drift']['false_positive_rate'] == 1
    assert high['concept_drift']['actual_negatives'] == 4
    assert low['delta']['prevalence'] == 0.25
    assert high['delta']['prevalence'] == -0.25


def test_unlabeled_evaluation_is_explicitly_unavailable(production):
    frame = production.drop(columns=TARGET)
    result = evaluate_concept(frame, frame.copy(), np.full(len(frame), 0.2), 0.27)
    assert result['status'] == 'unavailable_without_labels'
    assert result['control'] is result['concept_drift'] is result['delta'] is None
    assert result['segments'] == {}
    assert result['changed_labels'] is None


@pytest.mark.parametrize('case', ['feature', 'id_order', 'audit', 'mixed_labels', 'invalid_contract'])
def test_evaluation_rejects_unpaired_or_invalid_inputs(production, case):
    changed = build_concept_scenario(production)
    if case == 'feature':
        changed.loc[0, 'LIMIT_BAL'] += 1000
    elif case == 'id_order':
        changed = changed.iloc[::-1]
    elif case == 'audit':
        changed.loc[0, 'SEX'] = 2
    elif case == 'mixed_labels':
        changed = changed.drop(columns=TARGET)
    else:
        changed.loc[0, 'AGE'] = 18
    with pytest.raises((ValueError, InvalidBatch)):
        evaluate_concept(production, changed, np.full(len(production), .2), .27)


@pytest.mark.parametrize('scores', [[], [0.1], [float('nan')]*32, [-.1]*32, [1.1]*32, [[.2]]*32])
def test_evaluation_rejects_invalid_shared_probabilities(production, scores):
    with pytest.raises(ValueError):
        evaluate_concept(production, build_concept_scenario(production), scores, .27)


@pytest.mark.parametrize('threshold', [-.1, 1.1, float('nan'), float('inf')])
def test_evaluation_rejects_invalid_threshold(production, threshold):
    with pytest.raises(ValueError):
        evaluate_concept(production, production.copy(), np.full(32, .2), threshold)


def test_empty_segment_does_not_invent_metrics(production):
    frame = production.iloc[:16]
    result = evaluate_concept(frame, frame.copy(), np.full(16, .2), .27)
    assert result['segments']['pay_0_gt_0'] == {'status': 'empty_segment', 'rows': 0,
                                             'control': None, 'concept_drift': None, 'delta': None}


def test_concept_predictions_use_model_once_and_keep_scaler_coefficients_and_threshold(production, monkeypatch):
    model = train_models(production).model
    coef = model.named_steps['classifier'].coef_.copy()
    mean = model.named_steps['scale'].mean_.copy()
    expected = predict_scores(model, production.drop(columns=TARGET))
    original = model.predict_proba
    calls = []

    def predict(frame):
        calls.append(len(frame))
        return original(frame)

    def forbidden_fit(*args, **kwargs):
        pytest.fail('modelo congelado nao pode ser reajustado')

    monkeypatch.setattr(model, 'predict_proba', predict)
    monkeypatch.setattr(model, 'fit', forbidden_fit)
    predictions, evaluation = predict_concept(model, .27, production, build_concept_scenario(production))
    assert calls == [32]
    assert predictions.columns.tolist() == ['ID', 'probability_default', 'prediction']
    assert predictions.ID.tolist() == production.ID.tolist()
    np.testing.assert_array_equal(predictions.probability_default, expected)
    np.testing.assert_array_equal(predictions.prediction, (expected >= .27).astype(int))
    assert evaluation['threshold'] == .27
    np.testing.assert_array_equal(coef, model.named_steps['classifier'].coef_)
    np.testing.assert_array_equal(mean, model.named_steps['scale'].mean_)


@pytest.mark.parametrize('case', ['feature', 'mixed_labels', 'contract', 'threshold'])
def test_invalid_concept_pair_never_reaches_model(production, case):
    changed = build_concept_scenario(production)
    threshold = .27
    if case == 'feature':
        changed.loc[0, 'AGE'] += 1
    elif case == 'mixed_labels':
        changed = changed.drop(columns=TARGET)
    elif case == 'contract':
        changed.loc[0, 'AGE'] = 18
    else:
        threshold = float('nan')

    class ForbiddenModel:
        def predict_proba(self, frame):
            pytest.fail('par invalido chegou a inferencia')

    with pytest.raises((ValueError, InvalidBatch)):
        predict_concept(ForbiddenModel(), threshold, production, changed)


def test_frozen_prediction_supports_unlabeled_pair(production):
    model = train_models(production).model
    frame = production.drop(columns=TARGET)
    predictions, result = predict_concept(model, .27, frame, frame.copy())
    assert len(predictions) == len(frame)
    assert result['status'] == 'unavailable_without_labels'


@pytest.fixture
def inputs(production, tmp_path):
    data, model = tmp_path / 'data', tmp_path / 'model'
    data.mkdir()
    source = {'partitions': {}}
    for offset, name in enumerate(('reference', 'calibration', 'production')):
        frame = production.copy()
        frame['ID'] += offset * 100
        path = data / f'{name}.csv'
        frame.to_csv(path, index=False, lineterminator='\n')
        source['partitions'][name] = {'file': path.name, 'rows': len(frame),
                                     'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    payload = json.dumps(source).encode()
    (data / 'manifest.json').write_bytes(payload)
    save_models(train_models(production), model, thresholds={'model': .27, 'dummy': .5},
                metadata={'data_manifest_sha256': hashlib.sha256(payload).hexdigest()})
    return data, model


def test_concept_execution_reproduces_all_artifacts_without_changing_sources(inputs, tmp_path):
    data, model = inputs
    before = {p: p.read_bytes() for folder in inputs for p in folder.iterdir()}
    first, second = tmp_path / 'run1', tmp_path / 'run2'
    result = run_concept(data, model, first, trusted_model=True)
    run_concept(data, model, second, trusted_model=True)
    assert result['stage'] == 'S06'
    assert result['threshold'] == .27
    assert result['changed_labels'] == {'total': 8, 'zero_to_one': 4, 'one_to_zero': 4}
    assert {p.name for p in first.iterdir()} == {
        'control.csv', 'concept_drift.csv', 'predictions.csv', 'label_changes.csv',
        'report.json', 'manifest.json'}
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()
    for name, info in result['files'].items():
        assert hashlib.sha256((first / name).read_bytes()).hexdigest() == info['sha256']
    for path, content in before.items():
        assert path.read_bytes() == content
    changes = pd.read_csv(first / 'label_changes.csv')
    assert len(changes) == 8
    assert changes.columns.tolist() == ['ID', 'PAY_0', 'original_label', 'synthetic_label']
    report = json.loads((first / 'report.json').read_text())
    assert report['x_identical'] and report['shared_predictions']
    assert report['label_marginal_unchanged']
    assert 'ID' not in (first / 'report.json').read_text()


@pytest.mark.parametrize('case', ['production', 'reference', 'calibration', 'lineage', 'model'])
def test_concept_execution_rejects_corruption_before_success(inputs, tmp_path, case):
    data, model = inputs
    if case == 'lineage':
        path = model / 'manifest.json'
        value = json.loads(path.read_text())
        value['metadata']['data_manifest_sha256'] = 'another source'
        path.write_text(json.dumps(value))
    else:
        path = model / 'models.pkl' if case == 'model' else data / f'{case}.csv'
        path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError):
        run_concept(data, model, tmp_path / 'run', trusted_model=True)
    assert not (tmp_path / 'run/manifest.json').exists()


def test_concept_execution_rejects_overlapping_partition_ids(inputs, tmp_path):
    data, model = inputs
    path = data / 'production.csv'
    frame = pd.read_csv(path)
    frame.loc[0, 'ID'] = 1001
    frame.to_csv(path, index=False)
    value = json.loads((data / 'manifest.json').read_text())
    value['partitions']['production']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = json.dumps(value).encode()
    (data / 'manifest.json').write_bytes(payload)
    value = json.loads((model / 'manifest.json').read_text())
    value['metadata']['data_manifest_sha256'] = hashlib.sha256(payload).hexdigest()
    (model / 'manifest.json').write_text(json.dumps(value))
    with pytest.raises(ValueError, match='disjuntos'):
        run_concept(data, model, tmp_path / 'run', trusted_model=True)


def test_concept_execution_requires_trusted_local_model(inputs, tmp_path):
    with pytest.raises(ValueError, match='trusted'):
        run_concept(*inputs, tmp_path / 'run')
    assert not (tmp_path / 'run').exists()


def test_concept_execution_refuses_existing_directory(inputs, tmp_path):
    output = tmp_path / 'run'
    output.mkdir()
    marker = output / 'manifest.json'
    marker.write_text('keep')
    with pytest.raises(FileExistsError):
        run_concept(*inputs, output, trusted_model=True)
    assert marker.read_text() == 'keep'


def test_failed_concept_export_never_leaves_success_manifest(inputs, tmp_path, monkeypatch):
    original = Path.write_bytes

    def fail(path, payload):
        if path.name == 'predictions.csv':
            raise OSError('synthetic write failure')
        return original(path, payload)

    monkeypatch.setattr(Path, 'write_bytes', fail)
    with pytest.raises(OSError):
        run_concept(*inputs, tmp_path / 'run', trusted_model=True)
    assert not (tmp_path / 'run/manifest.json').exists()
