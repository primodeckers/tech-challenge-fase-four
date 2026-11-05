"""Executa um cenário do pipeline de monitoramento e imprime o estado final."""

import argparse
import json
from pathlib import Path

from credit_monitor.pipeline import SCENARIOS, run_pipeline

EXIT_CODES = {'succeeded': 0, 'failed': 1, 'rejected': 2, 'unavailable': 3}
TRACKING_FAILURE_EXIT = 4


def _tracking_receipt(run_dir, tracking_dir):
    try:
        from credit_monitor.tracking import track_run

        mlflow_run_id = track_run(run_dir, tracking_dir)
        receipt = {'status': 'recorded', 'mlflow_run_id': mlflow_run_id, 'reason': None}
    except Exception:
        receipt = {'status': 'failed', 'mlflow_run_id': None, 'reason': 'tracking_failed'}

    try:
        temporary = run_dir / '.tracking.tmp'
        temporary.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
        temporary.replace(run_dir / 'tracking.json')
    except OSError:
        receipt.update(status='failed', reason='tracking_receipt_failed')
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scenario', choices=SCENARIOS)
    parser.add_argument('--data-dir', type=Path, default=Path('data/processed'))
    parser.add_argument('--model-dir', type=Path, default=Path('artifacts/baseline'))
    parser.add_argument('--policy-dir', type=Path, default=Path('artifacts/drift'))
    parser.add_argument('--runs-dir', type=Path, default=Path('artifacts/pipeline'))
    parser.add_argument('--batch', type=Path, help='CSV alternativo à produção do manifesto')
    parser.add_argument('--tracking-dir', type=Path, help='Registra a execução no MLflow local')
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
    summary = {field: result[field] for field in fields}
    exit_code = EXIT_CODES[result['status']]
    if args.tracking_dir is not None:
        receipt = _tracking_receipt(args.runs_dir / result['run_id'], args.tracking_dir)
        summary['tracking'] = receipt
        if receipt['status'] == 'failed' and exit_code == 0:
            exit_code = TRACKING_FAILURE_EXIT
    print(json.dumps(summary))
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
