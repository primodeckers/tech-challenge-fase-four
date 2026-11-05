"""Executa um cenário do pipeline de monitoramento e imprime o estado final."""

import argparse
import json
from pathlib import Path

from credit_monitor.pipeline import SCENARIOS, run_pipeline

EXIT_CODES = {'succeeded': 0, 'failed': 1, 'rejected': 2, 'unavailable': 3}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scenario', choices=SCENARIOS)
    parser.add_argument('--data-dir', type=Path, default=Path('data/processed'))
    parser.add_argument('--model-dir', type=Path, default=Path('artifacts/baseline'))
    parser.add_argument('--policy-dir', type=Path, default=Path('artifacts/drift'))
    parser.add_argument('--runs-dir', type=Path, default=Path('artifacts/pipeline'))
    parser.add_argument('--batch', type=Path, help='CSV alternativo à produção do manifesto')
    parser.add_argument('--trusted-model', action='store_true',
                        help='Carregar apenas o modelo local de origem conhecida')
    args = parser.parse_args(argv)

    try:
        result = run_pipeline(
            scenario=args.scenario, data_dir=args.data_dir, model_dir=args.model_dir,
            policy_dir=args.policy_dir, runs_dir=args.runs_dir, batch_path=args.batch,
            trusted_model=args.trusted_model,
        )
    except OSError:
        print(json.dumps({'status': 'failed', 'scenario': args.scenario,
                          'run_id': None, 'reason': 'run_storage_failed'}))
        return EXIT_CODES['failed']

    fields = ('run_id', 'scenario', 'status', 'rows', 'failed_stage', 'reason', 'alert_count')
    print(json.dumps({field: result[field] for field in fields}))
    return EXIT_CODES[result['status']]


if __name__ == '__main__':
    raise SystemExit(main())
