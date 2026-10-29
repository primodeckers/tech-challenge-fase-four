import numpy as np
import pytest

from credit_monitor.drift import population_stability_index


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
