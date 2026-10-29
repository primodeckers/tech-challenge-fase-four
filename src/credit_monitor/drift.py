"""Estatistica de Data Drift com referencia fixa e estados explicitos."""

import numpy as np
from scipy.stats import ks_2samp


def ks_comparison(reference, current, *, min_samples=200):
    """KS bilateral; D e descritivo, p assintotico assume distribuicao continua."""
    _minimum(min_samples)
    x, y = _vector(reference), _vector(current)
    result = {"n_reference": len(x), "n_current": len(y), "statistic": None, "pvalue": None,
              "ties": bool(len(np.unique(np.r_[x, y])) < len(x) + len(y)),
              "pvalue_method": "asymptotic_continuous_approximation"}
    if min(len(x), len(y)) < min_samples:
        return {**result, "status": "insufficient_samples"}
    comparison = ks_2samp(x, y, alternative="two-sided", method="asymp", nan_policy="raise")
    return {**result, "status": "ok", "statistic": float(comparison.statistic),
            "pvalue": float(comparison.pvalue)}


def holm_adjust(pvalues):
    """Holm step-down para a familia de features avaliadas, sem descartar testes."""
    if any(not np.isfinite(p) or not 0 <= p <= 1 for p in pvalues.values()):
        raise ValueError("p-valores precisam estar entre0 e1")
    ordered = sorted(pvalues, key=lambda name: (pvalues[name], name))
    adjusted, previous = {}, 0.0
    for rank, name in enumerate(ordered):
        previous = max(previous, min(1.0, (len(ordered) - rank) * pvalues[name]))
        adjusted[name] = float(previous)
    return {name: adjusted[name] for name in pvalues}


def _vector(values):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("Amostra deve ser vetor numerico nao vazio e finito")
    return array


def _minimum(value):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("min_samples precisa ser inteiro positivo")


def _cuts(reference):
    if np.min(reference) == np.max(reference):
        return []
    quantiles = np.unique(np.quantile(reference, np.arange(1, 10) / 10))
    # nextafter inclui a massa exatamente no quantil no bin inferior (ex.: zeros).
    cuts = np.nextafter(quantiles[quantiles < np.max(reference)], np.inf)
    if not len(cuts):
        cuts = np.array([np.nextafter(np.min(reference), np.inf)])
    return cuts.tolist()


def population_stability_index(reference, current, *, cutpoints=None, min_samples=200, smoothing=0.5):
    """PSI com bordas infinitas e pseudocontagem por bin; nao decide alertas."""
    _minimum(min_samples)
    if not np.isfinite(smoothing) or smoothing <= 0:
        raise ValueError("Suavizacao precisa ser positiva e finita")
    x, y = _vector(reference), _vector(current)
    cuts = _cuts(x) if cutpoints is None else np.asarray(cutpoints, dtype=float).tolist()
    if np.asarray(cuts).ndim != 1 or not np.isfinite(cuts).all() or (np.diff(cuts) <= 0).any():
        raise ValueError("Cortes precisam ser finitos e estritamente crescentes")
    result = {"n_reference": len(x), "n_current": len(y), "psi": None,
              "cutpoints": cuts, "smoothing": float(smoothing)}
    if min(len(x), len(y)) < min_samples:
        return {**result, "status": "insufficient_samples"}
    if np.min(x) == np.max(x):
        status = "constant_reference_changed" if np.any(y != x[0]) else "constant_reference"
        return {**result, "status": status}
    edges = np.r_[-np.inf, cuts, np.inf]
    a, _ = np.histogram(x, edges)
    b, _ = np.histogram(y, edges)
    p = (a + smoothing) / (len(x) + smoothing * len(a))
    q = (b + smoothing) / (len(y) + smoothing * len(b))
    return {**result, "status": "ok", "psi": float(np.sum((q - p) * np.log(q / p))),
            "reference_counts": a.tolist(), "current_counts": b.tolist()}
