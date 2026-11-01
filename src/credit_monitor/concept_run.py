"""Execucao local do cenario S06, com entradas verificadas e modelo congelado."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from credit_monitor.baseline import FEATURES
from credit_monitor.concept import predict_concept
from credit_monitor.data import TARGET
from credit_monitor.model_store import load_models
from credit_monitor.scenario_run import _verified_partition
from credit_monitor.scenarios import build_concept_scenario


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')


def run_concept(data_dir, model_dir, output_dir, *, trusted_model=False, seed=42):
    """Exporta o par, predicoes compartilhadas e analise sem sobrescrever.

    CSVs individuais ficam locais. Manifesto e gravado por ultimo; sua ausencia
    indica execucao incompleta. O cenario e separado da intervencao em X da S04.
    """
    data_dir, model_dir, output_dir = Path(data_dir), Path(model_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError('Escolha uma pasta nova para preservar execucoes existentes')
    models, model_manifest = load_models(model_dir, trusted=trusted_model)
    manifest_bytes = (data_dir / 'manifest.json').read_bytes()
    source_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if model_manifest['metadata'].get('data_manifest_sha256') != source_sha:
        raise ValueError('Dados nao correspondem a origem registrada no artefato S03')
    source = json.loads(manifest_bytes)
    # Reusa a mesma verificacao de SHA, nome, contagem e contrato da S04.
    partitions = {name: _verified_partition(data_dir, name, source)
                  for name in ('reference', 'calibration', 'production')}
    names = list(partitions)
    for i, name in enumerate(names):
        for other in names[i+1:]:
            if set(partitions[name].ID) & set(partitions[other].ID):
                raise ValueError('IDs das tres particoes precisam ser disjuntos')
    control = partitions['production']
    concept = build_concept_scenario(control, seed=seed)
    threshold = model_manifest['thresholds']['model']
    predictions, report = predict_concept(models.model, threshold, control, concept)
    changed = control[TARGET].to_numpy() != concept[TARGET].to_numpy()
    changes = control.loc[changed, ['ID', 'PAY_0']].copy()
    changes['original_label'] = control.loc[changed, TARGET].to_numpy()
    changes['synthetic_label'] = concept.loc[changed, TARGET].to_numpy()
    known = pd.MultiIndex.from_frame(partitions['reference'].loc[:, FEATURES])
    shared = pd.MultiIndex.from_frame(control.loc[:, FEATURES]).isin(known)
    rule = {'feature': 'PAY_0', 'zero_to_one_condition': 'PAY_0 <= 0 and Y == 0',
            'one_to_zero_condition': 'PAY_0 > 0 and Y == 1',
            'per_direction_count': 'floor(min(eligible_zero_to_one, eligible_one_to_zero)/4)',
            'selection': 'sort candidates by ID; sample without replacement; zero_to_one first',
            'seed': seed, 'rng': 'numpy.random.PCG64', 'uses_model_errors': False}
    manifest = {
        'stage': 'S06', 'format_version': 1, 'rule': rule,
        'rows': len(control), 'threshold': threshold,
        'changed_labels': report['changed_labels'],
        'model_sha256': model_manifest['model_sha256'],
        'model_manifest_sha256': hashlib.sha256((model_dir / 'manifest.json').read_bytes()).hexdigest(),
        'data_manifest_sha256': source_sha,
        'input_sha256': {name: source['partitions'][name]['sha256'] for name in names},
        'versions': model_manifest['versions'],
        'code_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                        for name in ('concept_run.py', 'concept.py', 'scenarios.py', 'scenario_run.py',
                                     'baseline.py', 'contracts.py', 'model_store.py')},
        'production_profiles_seen_in_reference': int(shared.sum()),
        'invariants': {'x_identical': report['x_identical'],
                       'shared_predictions': report['shared_predictions'],
                       'label_marginal_unchanged': report['label_marginal_unchanged']},
        'limitations': [
            'Synthetic conditional label intervention, not evidence of real production drift or economic causality.',
            'Fixed balanced label swaps preserve global prevalence; conditional changes remain by construction.',
            'Metrics are descriptive for a paired synthetic scenario, not an independent generalization estimate.',
            'Production retains feature profiles seen in reference; exact profile independence is not guaranteed.',
            'Without labels, performance evaluation is unavailable; feature drift tests cannot establish concept drift.',
        ], 'files': {},
    }
    payloads = {name: frame.to_csv(index=False, lineterminator='\n').encode('utf-8')
                for name, frame in [('control.csv', control), ('concept_drift.csv', concept),
                                     ('predictions.csv', predictions), ('label_changes.csv', changes)]}
    payloads['report.json'] = _json_bytes(report)
    manifest['files'] = {name: {'sha256': hashlib.sha256(payload).hexdigest()}
                         for name, payload in payloads.items()}
    serialized = _json_bytes(manifest)
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, payload in payloads.items():
        (output_dir / name).write_bytes(payload)
    (output_dir / 'manifest.json').write_bytes(serialized)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/processed'))
    parser.add_argument('--model-dir', type=Path, default=Path('artifacts/baseline-s03-20261026'))
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/concept-s06'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trusted-model', action='store_true', help='Somente artefato local de origem conhecida')
    args = parser.parse_args()
    manifest = run_concept(args.data_dir, args.model_dir, args.output_dir,
                           trusted_model=args.trusted_model, seed=args.seed)
    print(json.dumps({'output_dir': str(args.output_dir), 'rows': manifest['rows'],
                      'threshold': manifest['threshold'], 'changed_labels': manifest['changed_labels'],
                      'invariants': manifest['invariants']}, indent=2))


if __name__ == '__main__':
    main()
