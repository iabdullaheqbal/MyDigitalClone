import pytest

from clone.retrieval.fusion import rrf


def test_rrf_values_use_one_based_ranks() -> None:
    fused = dict(rrf([[4, 2], [5, 4, 2]], k=60))

    assert fused[4] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[2] == pytest.approx(1 / 62 + 1 / 63)
    assert fused[5] == pytest.approx(1 / 61)


def test_rrf_is_stable_for_ties_and_supports_single_or_disjoint_lists() -> None:
    assert rrf([[7], [8]], k=60) == [(7, 1 / 61), (8, 1 / 61)]
    assert rrf([[3, 1]], k=60) == [(3, 1 / 61), (1, 1 / 62)]
    assert rrf([[2, 2]], k=60) == [(2, 1 / 61)]


def test_rrf_rejects_an_invalid_constant() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        rrf([[1]], k=0)
