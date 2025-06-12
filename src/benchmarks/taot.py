# TAOT benchmark
import numpy as np
import ot

def taot_discrepancy(A, B, lambd=100, w=1.0, max_iter=5000, tol=1e-3):
    """
    Compute Time Adaptive Optimal Transport (TAOT) distance between two time series A and B.
    
    Parameters:
        A, B (np.ndarray): 1D arrays of the same length representing time series.
        lambda_reg (float): Entropic regularization coefficient.
        w (float): Weight for time difference.
        max_iter (int): Maximum number of Sinkhorn iterations.
        tol (float): Convergence tolerance.
        
    Returns:
        distance (float): The TAOT distance.
        transport_plan (np.ndarray): The optimal transport matrix (many-to-many alignment).
    """
    assert len(A) == len(B), "Time series must be of equal length"
    n = len(A)

    # Normalize time indices
    t = (np.arange(n) - np.mean(np.arange(n))) / np.std(np.arange(n))

    M = ot.dist(A.reshape(-1, 1), B.reshape(-1, 1), metric='sqeuclidean') + w * ot.dist(t.reshape(-1, 1), t.reshape(-1, 1), metric='sqeuclidean')

    # Normalize cost matrix
    M /= np.median(M)

    # Define uniform distributions
    p = np.ones(n) / n
    q = np.ones(n) / n

    # Compute Sinkhorn distance
    transport_plan = ot.sinkhorn(p, q, M, reg=lambd, numItermax=max_iter, stopThr=tol)
    distance = np.sum(transport_plan * M)

    return distance

if __name__ == "__main__":
    # Example usage
    A = np.random.rand(5)  # Example time series A
    B = np.random.rand(5)  # Example time series B
    distance = taot_discrepancy(A, B)
    print("TAOT Distance:", distance)