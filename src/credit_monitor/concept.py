"""Avaliacao pareada de uma intervencao sintetica no alvo, com X fixo."""

import numpy as np

from credit_monitor.baseline import evaluate_binary
from credit_monitor.data import TARGET
from credit_monitor.scenarios import copy_control


def _metrics(labels, scores, threshold):
    metrics = evaluate_binary(labels, scores, threshold)
    tn, fp = metrics['confusion_matrix'][0]
    fn, tp = metrics['confusion_matrix'][1]
    metrics.update({
        'actual_positives': tp + fn, 'actual_negatives': tn + fp,
        'accuracy': (tn + tp) / len(labels),
        'false_positive_rate': fp / (fp + tn) if fp + tn else None,
        'false_negative_rate': fn / (fn + tp) if fn + tp else None,
    })
    return metrics


def _delta(first, second):
    names = ('prevalence', 'roc_auc', 'average_precision', 'precision', 'recall',
             'f1', 'brier', 'accuracy', 'false_positive_rate', 'false_negative_rate')
    return {name: second[name] - first[name]
            if first[name] is not None and second[name] is not None else None
            for name in names}


def evaluate_concept(control, concept, scores, threshold):
    """Compara dois alvos no mesmo X, com um unico vetor de probabilidades.

    Nao treina, prediz ou escolhe limiar. Erros por PAY_0 trazem denominadores;
    grupos vazios/classes ausentes e ausencia de rotulos sao explicitos.
    Resultado descritivo da intervencao, sem teste de drift real ou causalidade.
    """
    control, concept = copy_control(control), copy_control(concept)
    labeled = TARGET in control
    if labeled != (TARGET in concept):
        raise ValueError('Os dois lotes precisam da mesma disponibilidade de rotulos')
    columns = [name for name in control if name != TARGET]
    if not control[columns].reset_index(drop=True).equals(concept[columns].reset_index(drop=True)):
        raise ValueError('Comparacao de conceito requer X, IDs e ordem identicos')
    scores = np.asarray(scores, dtype=float)
    if (scores.shape != (len(control),) or not np.isfinite(scores).all()
            or ((scores < 0) | (scores > 1)).any()):
        raise ValueError('Probabilidades precisam ser vetor finito entre0 e1 pareado ao lote')
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Limiar precisa ser finito entre0 e1')
    result = {'status': 'available' if labeled else 'unavailable_without_labels',
              'rows': len(control), 'threshold': float(threshold),
              'x_identical': True, 'shared_predictions': True,
              'control': None, 'concept_drift': None, 'delta': None,
              'segments': {}, 'changed_labels': None, 'label_marginal_unchanged': None}
    if not labeled:
        return result
    first, second = control[TARGET].to_numpy(), concept[TARGET].to_numpy()
    a, b = _metrics(first, scores, threshold), _metrics(second, scores, threshold)
    result.update({'control': a, 'concept_drift': b, 'delta': _delta(a, b),
                   'changed_labels': {'total': int((first != second).sum()),
                                      'zero_to_one': int(((first == 0) & (second == 1)).sum()),
                                      'one_to_zero': int(((first == 1) & (second == 0)).sum())},
                   'label_marginal_unchanged': bool(first.sum() == second.sum())})
    for name, mask in [('pay_0_le_0', control.PAY_0.to_numpy() <= 0),
                       ('pay_0_gt_0', control.PAY_0.to_numpy() > 0)]:
        count = int(mask.sum())
        segment = {'status': 'available' if count else 'empty_segment', 'rows': count,
                   'control': None, 'concept_drift': None, 'delta': None}
        if count:
            a = _metrics(first[mask], scores[mask], threshold)
            b = _metrics(second[mask], scores[mask], threshold)
            segment.update({'control': a, 'concept_drift': b, 'delta': _delta(a, b)})
        result['segments'][name] = segment
    return result
