
import warnings
from itertools import product

import numpy as np
import pytest

import ot
from ot.backend import tf, torch
from tmw.sinkhorn import tmw_sinkhorn, get_mask, tmw_sinkhorn_log


@pytest.mark.parametrize("verbose, warn", product([True, False], [True, False]))
def test_sinkhorn(verbose, warn):
    # test sinkhorn
    n = 100
    rng = np.random.RandomState(0)
    
    mask = get_mask(n, n, 5)  # locality constraint
    x = rng.randn(n, 2)
    u = ot.utils.unif(n)

    M = ot.dist(x, x)

    G = tmw_sinkhorn(u, u, M, mask, 1, stopThr=1e-10, verbose=verbose, warn=warn)

    # check constraints
    np.testing.assert_allclose(u, G.sum(1), atol=1e-05)  # cf convergence sinkhorn
    np.testing.assert_allclose(u, G.sum(0), atol=1e-05)  # cf convergence sinkhorn

    with pytest.warns(UserWarning):
        tmw_sinkhorn(u, u, M, mask, 1, stopThr=0, numItermax=1)


@pytest.mark.parametrize(
    "method",
    [
        "tmw_sinkhorn",
        "tmw_sinkhorn_log",
    ],
)
def test_convergence_warning(method):
    # test sinkhorn
    n = 100
    a1 = ot.datasets.make_1D_gauss(n, m=30, s=10)
    a2 = ot.datasets.make_1D_gauss(n, m=40, s=10)
    A = np.asarray([a1, a2]).T
    M = ot.utils.dist0(n)
    mask = get_mask(n, n, 5)  # locality constraint
    with pytest.warns(UserWarning):
        ot.sinkhorn(a1, a2, M, 1.0, method=method, stopThr=0, numItermax=1)

    if method in ["sinkhorn", "sinkhorn_log"]:
        with pytest.warns(UserWarning):
            ot.barycenter(A, M, 1, method=method, stopThr=0, numItermax=1)
        with pytest.warns(UserWarning):
            ot.sinkhorn2(
                a1, a2, M, 1, method=method, stopThr=0, numItermax=1, warn=True
            )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ot.sinkhorn2(
                a1, a2, M, 1, method=method, stopThr=0, numItermax=1, warn=False
            )


@pytest.mark.parametrize("method", ["sinkhorn", "sinkhorn_stabilized"])
def test_nan_warning(method):
    # test sinkhorn
    n = 100
    a1 = ot.datasets.make_1D_gauss(n, m=30, s=10)
    a2 = ot.datasets.make_1D_gauss(n, m=40, s=10)

    M = ot.utils.dist0(n)
    reg = 0
    with pytest.warns(UserWarning):
        # warn set to False to avoid catching a convergence warning instead
        ot.sinkhorn(a1, a2, M, reg, method=method, warn=False)


def test_sinkhorn_stabilization():
    # test sinkhorn
    n = 100
    a1 = ot.datasets.make_1D_gauss(n, m=30, s=10)
    a2 = ot.datasets.make_1D_gauss(n, m=40, s=10)
    M = ot.utils.dist0(n)
    reg = 1e-5
    loss1 = ot.sinkhorn2(a1, a2, M, reg, method="sinkhorn_log")
    loss2 = ot.sinkhorn2(a1, a2, M, reg, tau=1, method="sinkhorn_stabilized")
    np.testing.assert_allclose(loss1, loss2, atol=1e-06)  # cf convergence sinkhorn

