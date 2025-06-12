# OTW benchmark
import numpy as np

def otw_discrepancy(a: np.ndarray, b: np.ndarray, m: float = 1.0, s: int = 10) -> float:
    """
    Compute OTW_{m,s}(a, b) as defined in Equation (8) of the OTW paper.

    Parameters:
        a (np.ndarray): First time series, shape (n,)
        b (np.ndarray): Second time series, shape (n,)
        m (float): Waste cost parameter
        s (int): Window size (localness parameter), must be in [1, n]

    Returns:
        float: OTW_{m,s}(a, b) distance
    """
    assert a.shape == b.shape, "Time series must have the same length"
    assert 1 <= s <= len(a), "Window size s must be between 1 and length of time series"

    def windowed_cumsum(x, s):
        # Compute windowed cumulative sum A_s(i) for each i
        cumsum = np.cumsum(x)
        padded = np.pad(cumsum, (s, 0), mode='constant', constant_values=0)
        return cumsum - padded[:-s]

    A_s = windowed_cumsum(a, s)
    B_s = windowed_cumsum(b, s)
    
    distance = m * abs(A_s[-1] - B_s[-1]) + np.sum(np.abs(A_s[:-1] - B_s[:-1]))
    return distance

def main():
    # Example usage
    import timeit
    a = np.random.random_sample((100,))
    b = np.random.random_sample((100,))
    m = 1.0
    s = 10
    start_time = timeit.default_timer()
    distance = otw_discrepancy(a, b, m, s)
    end_time = timeit.default_timer()
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"OTW_m,s distance: {distance}")
if __name__ == "__main__":  
    main()