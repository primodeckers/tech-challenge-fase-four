"""Estatistica de Data Drift com referencia fixa e estados explicitos."""

import hashlib

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from credit_monitor.baseline import FEATURES
from credit_monitor.contracts import validate_batch
from credit_monitor.data import TARGET


def _frame(frame):
    mode = "training" if isinstance(frame, pd.DataFrame) and TARGET in frame else "inference"
    return validate_batch(frame, mode=mode)


def _reference_digest(frame):
    payload = frame.loc[:, ["ID", *FEATURES]].sort_values("ID").to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode()).hexdigest()


def calibrate_policy(reference, calibration, *, n_resamples=200, seed=42, quantile=0.99, min_samples=200, alpha=0.05):
    """Envelopes empiricos de PSI/D por reamostragem da calibracao, nao p-valores.

    Referencia permanece fixa. Nao recebe nem consulta dados de producao.
    """
    _minimum(min_samples)
    if not isinstance(n_resamples, int) or isinstance(n_resamples, bool) or n_resamples < 20:
        raise ValueError("Use pelo menos20 reamostragens")
    if not 0 < quantile < 1 or not 0 < alpha < 1:
        raise ValueError("quantile e alpha precisam estar entre0 e1")
    reference, calibration = _frame(reference), _frame(calibration)
    if set(reference.ID) & set(calibration.ID):
        raise ValueError("IDs de referencia e calibracao devem ser disjuntos")
    if min(len(reference), len(calibration)) < min_samples:
        raise ValueError("Amostra insuficiente para calibrar politica")
    rng = np.random.default_rng(seed)
    policy = {"format_version": 1, "reference_sha256": _reference_digest(reference),
              "reference_rows": len(reference), "calibration_rows": len(calibration),
              "min_samples": min_samples, "n_resamples": n_resamples, "seed": seed,
              "quantile": quantile, "alpha": alpha, "smoothing": 0.5,
              "method": "calibration_bootstrap_envelope_not_confidence_bound", "features": {}}
    for column in FEATURES:
        x, y = _vector(reference[column]), _vector(calibration[column])
        cuts = _cuts(x)
        if np.min(x) == np.max(x):
            policy["features"][column] = {"status": "constant_reference", "cutpoints": [],
                "psi_threshold": None, "ks_threshold": None}
            continue
        psi_values, distances = [], []
        for _ in range(n_resamples):
            sampled = rng.choice(y, size=len(y), replace=True)
            psi_values.append(population_stability_index(x, sampled, cutpoints=cuts,
                                                         min_samples=min_samples)["psi"])
            distances.append(ks_2samp(x, sampled, method="asymp").statistic)
        policy["features"][column] = {"status": "ok", "cutpoints": cuts,
            "psi_threshold": float(np.quantile(psi_values, quantile)),
            "ks_threshold": float(np.quantile(distances, quantile))}
    return policy


def compare_drift(reference, current, policy):
    """Compara um lote usando politica ja definida, sem recalibrar limiares."""
    reference, current = _frame(reference), _frame(current)
    if policy["format_version"] != 1 or set(policy["features"]) != set(FEATURES):
        raise ValueError("Formato/features da politica incompativeis")
    if _reference_digest(reference) != policy["reference_sha256"]:
        raise ValueError("Politica nao pertence a esta referencia")
    minimum = policy["min_samples"]
    _minimum(minimum)
    if not 0 < policy["alpha"] < 1:
        raise ValueError("Alpha invalido na politica")
    status = "ok"
    if min(len(reference), len(current)) < minimum:
        status = "insufficient_samples"
    elif len(current) != policy["calibration_rows"]:
        status = "uncalibrated_sample_size"
    features, pvalues = {}, {}
    for column in FEATURES:
        limits = policy["features"][column]
        if limits["status"] == "ok" and any(
            not np.isfinite(limits[key]) or limits[key] < 0 for key in ("psi_threshold", "ks_threshold")
        ):
            raise ValueError("Limiares invalidos na politica")
        psi = population_stability_index(reference[column], current[column], cutpoints=limits["cutpoints"],
                                        min_samples=minimum, smoothing=policy["smoothing"])
        ks = ks_comparison(reference[column], current[column], min_samples=minimum)
        features[column] = {"status": psi["status"], "psi": psi["psi"],
                            "psi_threshold": limits["psi_threshold"], "ks_threshold": limits["ks_threshold"],
                            "ks": ks, "alert": None, "ks_pvalue_holm": None}
        if ks["pvalue"] is not None:
            pvalues[column] = ks["pvalue"]
    adjusted = holm_adjust(pvalues)
    for column, info in features.items():
        info["ks_pvalue_holm"] = adjusted.get(column)
        if status != "ok":
            continue
        if info["status"] == "constant_reference_changed":
            info["alert"] = True
            info["reason"] = "constant_support_changed"
        elif info["status"] == "constant_reference":
            info["alert"] = False
            info["reason"] = "constant_unchanged"
        else:
            info["alert"] = bool(info["psi"] > info["psi_threshold"]
                and info["ks"]["statistic"] > info["ks_threshold"]
                and info["ks_pvalue_holm"] <= policy["alpha"])
            info["reason"] = "psi_and_ks" if info["alert"] else "no_alert"
    alerts = [column for column, info in features.items() if info["alert"] is True]
    return {"status": status, "n_reference": len(reference), "n_current": len(current),
            "drift_detected": bool(alerts) if status == "ok" else None,
            "alerted_features": alerts, "correction": "Holm across evaluated features per comparison",
            "features": features}


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
