import numpy as np
import pytest
from tmw.network_simplex import tmw_network_simplex, banded_ot_ortools

def test_tmw_network_simplex_basic():
    # Simple test: identical arrays, bandwidth covers all
    x = np.arange(10)
    y = np.arange(10)
    dist = tmw_network_simplex(x, y, bandwidth=9)
    assert np.isclose(dist, 0), f"Expected 0, got {dist}"

def test_tmw_network_simplex_bandwidth():
    # Test: different arrays, small bandwidth
    x = np.arange(10)
    y = np.arange(10)[::-1]
    dist = tmw_network_simplex(x, y, bandwidth=1)
    assert dist > 0

def test_banded_ot_ortools_basic():
    # Simple test: identical arrays, should match exactly
    x = np.arange(5)
    y = np.arange(5)
    cost, matching = banded_ot_ortools(x, y, w=4)
    assert np.isclose(cost, 0), f"Expected 0, got {cost}"
    assert len(matching) == 5

@pytest.mark.parametrize("w", [0, 1, 2, 4])
def test_banded_ot_ortools_bandwidth(w):
    x = np.arange(6)
    y = np.arange(6)[::-1]
    cost, matching = banded_ot_ortools(x, y, w=w)
    assert cost >= 0
    assert len(matching) == 6

def test_tmw_network_simplex_assertion():
    # Should raise if input lengths differ
    x = np.arange(5)
    y = np.arange(6)
    with pytest.raises(AssertionError):
        tmw_network_simplex(x, y, bandwidth=1)
