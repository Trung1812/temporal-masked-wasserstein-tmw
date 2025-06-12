# DTW benchmark
import numpy as np
import torch
from torch.autograd import Function
from dtw import dtw

def dtw_discrepancy(x, y, dist_fn="euclidean"):
    """
    Computes the DTW distance between two sequences x and y.

    Parameters:
    ----------
    x : array-like
        First sequence.
    y : array-like
        Second sequence.
    dist_fn : callable, optional
        Function to compute the distance between two elements. If None, uses Euclidean distance.

    Returns:
    -------
    float
        The DTW distance between x and y.
    """
    dtw_output = dtw(x, y, keep_internals=False, distance_only=True, dist_method=dist_fn)
    return dtw_output.distance


##
##  The code for soft-DTW is adapted from https://github.com/Sleepwalking/pytorch-softdtw
##

def main():
    # Example usage
    x = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([[1, 2], [3, 4], [5, 7]])
    distance = dtw_discrepancy(x, y)
    print("DTW distance:", distance)

if __name__ == "__main__":
    main()