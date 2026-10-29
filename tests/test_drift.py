import numpy as np
import pytest

from credit_monitor.drift import holm_adjust, ks_comparison, population_stability_index


def test_psi_is_zero_for_identical_samples():
    values = np.arange(400)
    result = population_stability_index(values, values)
    assert result["status"] == "ok"
    assert result["psi"] == 0
    assert sum(result["reference_counts"]) == 400


def test_psi_matches_hand_calculation_with_fixed_bins_and_pseudocounts():
    result = population_stability_index([0, 0, 1, 1], [0, 1, 1, 1],
                                       cutpoints=[0.5], min_samples=1)
    # Smoothing0.5: p=(0.5,0.5), q=(0.3,0.7).
    assert result["psi"] == pytest.approx(0.2 * np.log(7 / 3))
    assert result["reference_counts"] == [2, 2]
    assert result["current_counts"] == [1, 3]


def test_psi_covers_extremes_and_zero_frequency_bins():
    result = population_stability_index([0, 0, 1, 1], [-10000, 10000, 10000, 10000],
                                       cutpoints=[-1, 0.5, 2], min_samples=1)
    assert sum(result["current_counts"]) == 4
    assert result["current_counts"] == [1, 0, 0, 3]
    assert np.isfinite(result["psi"]) and result["psi"] > 0


def test_quantile_cuts_separate_zero_mass_from_positive_values():
    reference = [0] * 390 + [1] * 10
    current = [0] * 200 + [1] * 200
    result = population_stability_index(reference, current)
    assert result["psi"] > 1
    assert len(result["reference_counts"]) >= 2


@pytest.mark.parametrize("bad", [[1, float('nan')], [1, float('inf')], [[1], [2]], []])
def test_psi_rejects_invalid_vectors_instead_of_dropping_values(bad):
    with pytest.raises(ValueError):
        population_stability_index(bad, [1, 2], min_samples=1)


def test_psi_reports_small_samples_and_constants_explicitly():
    small = population_stability_index([0, 1], [0, 2])
    assert small["status"] == "insufficient_samples" and small["psi"] is None
    constant = population_stability_index([1] * 200, [1] * 200)
    assert constant["status"] == "constant_reference" and constant["psi"] is None
    changed = population_stability_index([1] * 200, [2] * 200)
    assert changed["status"] == "constant_reference_changed" and changed["psi"] is None


@pytest.mark.parametrize("kwargs", [{"smoothing": 0}, {"smoothing": float('nan')},
    {"min_samples": 0}, {"cutpoints": [1, 1]}, {"cutpoints": [2, 1]}, {"cutpoints": [float('inf')]}])
def test_psi_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        population_stability_index(np.arange(400), np.arange(400), **kwargs)


def test_ks_identical_and_disjoint_samples_report_effect_and_pvalue():
    same = ks_comparison(np.arange(400), np.arange(400))
    assert same["statistic"] == 0 and same["pvalue"] == 1
    shifted = ks_comparison(np.arange(400), np.arange(400) + 1000)
    assert shifted["statistic"] == 1 and shifted["pvalue"] < 0.001
    assert shifted["n_reference"] == shifted["n_current"] == 400


def test_ks_reports_ties_and_small_samples_without_hiding_them():
    tied = ks_comparison([0] * 300, [1] * 300)
    assert tied["ties"] is True
    assert tied["pvalue_method"] == "asymptotic_continuous_approximation"
    small = ks_comparison([0, 1], [0, 2])
    assert small["status"] == "insufficient_samples"
    assert small["statistic"] is None and small["pvalue"] is None
    with pytest.raises(ValueError):
        ks_comparison([float('nan')], [1])


def test_holm_adjustment_matches_hand_calculation_and_preserves_names():
    assert holm_adjust({"c": 0.03, "a": 0.01, "b": 0.02}) == {
        "c": pytest.approx(0.04), "a": pytest.approx(0.03), "b": pytest.approx(0.04)}
    assert holm_adjust({"a": 0.8, "b": 0.9}) == {"a": 1.0, "b": 1.0}
    assert holm_adjust({}) == {}


@pytest.mark.parametrize("p", [-0.1, 1.1, float('nan')])
def test_holm_rejects_invalid_pvalues(p):
    with pytest.raises(ValueError):
        holm_adjust({"feature": p})
