"""Auditoria agregada do controle histórico, com modelo e limiar congelados."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from credit_monitor.baseline import FEATURES, evaluate_binary, predict_scores
from credit_monitor.contracts import validate_batch
from credit_monitor.data import TARGET
from credit_monitor.model_store import load_models
from credit_monitor.scenario_run import _verified_partition

AGE_BANDS = ((19, 29), (30, 44), (45, 59), (60, 100))
MIN_GROUP_ROWS = 30
MIN_RATE_DENOMINATOR = 20
GAP_METRICS = ('predicted_positive_rate', 'recall', 'false_positive_rate')


def _group_metrics(labels, scores, threshold):
    if len(labels) < MIN_GROUP_ROWS:
        return {'status': 'suppressed_small_group'}
    result = evaluate_binary(labels, scores, threshold)
    (tn, fp), (fn, tp) = result['confusion_matrix']
    observed_positive, observed_negative = tp + fn, tn + fp
    enough_positives = observed_positive >= MIN_RATE_DENOMINATOR
    return {
        'status': 'available', 'rows': len(labels),
        'observed_positive_count': observed_positive, 'observed_negative_count': observed_negative,
        'prevalence': result['prevalence'], 'predicted_positive_rate': result['predicted_positive_rate'],
        'recall': tp / observed_positive if enough_positives else None,
        'false_negative_rate': fn / observed_positive if enough_positives else None,
        'false_positive_rate': fp / observed_negative if observed_negative >= MIN_RATE_DENOMINATOR else None,
        'brier': result['brier'],
    }


def _gap(groups, metric):
    values = [group[metric] for group in groups.values() if group.get(metric) is not None]
    return 100 * (max(values) - min(values)) if len(values) >= 2 else None


def audit_groups(frame, scores, threshold):
    """Resume erros e denominadores por código de SEX e faixa de AGE, separadamente."""
    frame = validate_batch(frame, mode='training')
    scores = np.asarray(scores, dtype=float)
    if (scores.shape != (len(frame),) or not np.isfinite(scores).all()
            or ((scores < 0) | (scores > 1)).any()):
        raise ValueError('Probabilidades inválidas ou sem pareamento com o lote')
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Limiar inválido')
    ages = frame.AGE.to_numpy()
    sex_codes = frame.SEX.to_numpy()
    labels = frame[TARGET].to_numpy()
    masks = {
        'SEX': {str(code): sex_codes == code for code in (1, 2)},
        'AGE': {f'{low}-{high}': (ages >= low) & (ages <= high) for low, high in AGE_BANDS},
    }
    groups = {column: {name: _group_metrics(labels[mask], scores[mask], threshold)
                       for name, mask in by_group.items()} for column, by_group in masks.items()}
    return {
        'rows': len(frame), 'threshold': float(threshold), 'model_features': list(FEATURES),
        'min_group_rows': MIN_GROUP_ROWS, 'min_rate_denominator': MIN_RATE_DENOMINATOR,
        'groups': groups,
        'gaps_percentage_points': {column: {metric: _gap(by_group, metric) for metric in GAP_METRICS}
                                   for column, by_group in groups.items()},
    }


def run_audit(data_dir, model_dir, output_dir, *, trusted_model=False):
    """Exporta somente agregados, em uma pasta nova, para o controle original."""
    data_dir, model_dir, output_dir = map(Path, (data_dir, model_dir, output_dir))
    if output_dir.exists():
        raise FileExistsError('Escolha uma pasta nova para a auditoria')
    payload = (data_dir / 'manifest.json').read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    source = json.loads(payload)
    models, model_manifest = load_models(model_dir, trusted=trusted_model)
    if model_manifest['metadata'].get('data_manifest_sha256') != digest:
        raise ValueError('Origem do modelo não corresponde aos dados')
    reference = _verified_partition(data_dir, 'reference', source)
    production = _verified_partition(data_dir, 'production', source)
    if TARGET not in production or production.ID.isin(reference.ID).any():
        raise ValueError('Auditoria exige produção rotulada e IDs disjuntos da referência')
    scores = predict_scores(models.model, production.drop(columns=TARGET))
    report = audit_groups(production, scores, model_manifest['thresholds']['model'])
    report.update(stage='S09', format_version=1, scope='historical_control',
                  data_manifest_sha256=digest, production_sha256=source['partitions']['production']['sha256'],
                  model_sha256=model_manifest['model_sha256'], versions=model_manifest['versions'],
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    serialized = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + '\n'
    output_dir.mkdir(parents=True, exist_ok=False)
    temporary = output_dir / '.audit.tmp'
    temporary.write_text(serialized, encoding='utf-8', newline='\n')
    temporary.replace(output_dir / 'audit.json')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/processed'))
    parser.add_argument('--model-dir', type=Path, default=Path('artifacts/baseline'))
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/group-audit'))
    parser.add_argument('--trusted-model', action='store_true')
    args = parser.parse_args()
    report = run_audit(args.data_dir, args.model_dir, args.output_dir, trusted_model=args.trusted_model)
    print(json.dumps({'output_dir': str(args.output_dir), 'rows': report['rows'],
                      'threshold': report['threshold']}))


if __name__ == '__main__':
    main()
