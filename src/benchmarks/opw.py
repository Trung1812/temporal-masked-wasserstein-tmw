# OPW benchmark
import math

import numpy as np
import ot

from .utils import get_E_F


def opw_discrepancy(x, y, lambda1=50, lambda2=0.1, delta=1):
    """Compute the OPW discrepancy between two distributions.
    Args:
        M (ndarray): Cost matrix.
        lambda1 (float, optional): Weight of the first term. Defaults to 50.
        lambda2 (float, optional): Weight of the second term. Defaults to 0.1.
        delta (float, optional): Bandwidth parameter. Defaults to 1.
    Returns:
        float: The OPW discrepancy.
    """
    # If the input is 2 (n, )   # then we need to convert it to (n, 1)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    if y.ndim == 1:
        y = y[:, np.newaxis]
    M = ot.dist(x, y, metric="euclidean")
    rows, cols = M.shape

    a = np.ones(rows, dtype=M.dtype) / rows
    b = np.ones(cols, dtype=M.dtype) / cols
    E, F = get_E_F(a.shape[0], b.shape[0], backend=np)
    M_hat = (
        M
        - lambda1 * E
        + lambda2 * (F / (2 * delta**2) + np.log(delta * np.sqrt(2 * math.pi)))
    )
    # return ot.sinkhorn2(a, b, M_hat, reg=lambda2)
    return ot.emd2(a, b, M_hat)

def main():
    # Example usage
    import timeit
    x = np.random.random_sample((100, 2))
    y = np.random.random_sample((100, 2))
    M = ot.dist(x, y, metric="euclidean")
    lambda1 = 10
    lambda2 = 0.1
    delta = 1
    start_time = timeit.default_timer()
    discrepancy = opw_discrepancy(x, y, lambda1, lambda2, delta)
    end_time = timeit.default_timer()
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"OPW Discrepancy: {discrepancy}")
if __name__ == "__main__":
    main()