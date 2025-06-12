# exact OT solver via LEMON or OR-tools
import numpy as np
import networkx as nx
import numpy as np

from scipy.optimize import linear_sum_assignment
from ot.utils import dist
from ortools.graph.python import min_cost_flow as pywrapgraph

def tmw_network_simplex(u_values, v_values,
                          bandwidth=1,
                          metric='euclidean', p=1):
    assert u_values.shape[0] == v_values.shape[0], f"Expect array of the same length, get {u_values.shape[0]} and {v_values.shape[0]}"
    assert p >= 1, f"The OT loss is only valid for p >= 1, get {p}"
    n = u_values.shape[0]
    # Compute the cost matrix
    M = dist(np.expand_dims(u_values, axis=1), np.expand_dims(v_values, axis=1), metric=metric, p=p)
    # Apply the mask to the cost matrix
    row_idx, col_idx = np.ogrid[:n, :n]  # Create row and column index grids
    mask = np.abs(row_idx - col_idx) <= bandwidth  # Boolean mask for banded region

    M[mask == 0] = np.inf
    # Solve the linear sum assignment problem
    row_ind, col_ind = linear_sum_assignment(M)

    # Compute the Wasserstein distance
    distance = M[row_ind, col_ind].sum() / n

    return distance

def banded_ot_ortools(x, y, w, cost_fn=None):
    n = len(x)
    if cost_fn is None:
        cost_fn = lambda xi, yj: int(1e6 * abs(xi - yj))  # integer cost required

    smcf = pywrapgraph.SimpleMinCostFlow()

    # Node indexing:
    # Source = 2n, Sink = 2n + 1
    source = 2 * n
    sink = 2 * n + 1

    # Add edges from source to x-nodes
    for i in range(n):
        smcf.add_arc_with_capacity_and_unit_cost(source, i, 1, 0)

    # Add edges from y-nodes to sink
    for j in range(n):
        smcf.add_arc_with_capacity_and_unit_cost(n + j, sink, 1, 0)

    # Add locality-constrained edges from x_i to y_j
    for i in range(n):
        for j in range(max(0, i - w), min(n, i + w + 1)):
            cost = cost_fn(x[i], y[j])
            smcf.add_arc_with_capacity_and_unit_cost(i, n + j, 1, int(cost))

    # Node supplies
    supplies = [0] * (2 * n + 2)
    supplies[source] = n
    supplies[sink] = -n

    for node, supply in enumerate(supplies):
        smcf.set_node_supply(node, supply)

    # Solve
    status = smcf.solve()

    if status != smcf.OPTIMAL:
        raise RuntimeError("Min-cost flow did not solve optimally.")

    total_cost = smcf.optimal_cost() / 1e6  # undo cost scaling
    matching = []

    for i in range(smcf.num_arcs()):
        if smcf.flow(i) > 0:
            u, v = smcf.tail(i), smcf.head(i)
            if 0 <= u < n and n <= v < 2 * n:
                matching.append((u, v - n))

    return total_cost, matching

def main():
    import time
    from utils import log_time
    time_series_lengh = [20, 200, 2000, 20000]
    #for each length, create 2 random time series, 
    #then compute the time it takes to compute the 
    #distance between them, 

    num_samples = 10
    
    for length in time_series_lengh:
        total_time = 0
        for i in range(num_samples):
            x = np.random.rand(length)
            y = np.random.rand(length)
            w = 5
            start_time = time.time()
            cost = (x, y, w)
            end_time = time.time()
            total_time += end_time - start_time
        avg_time = total_time / num_samples
        print(f"Length: {length}, Average Time: {avg_time:.4f} seconds, Cost: {cost}")
        log_time(avg_time, length, "wasserstein_1d_metric_banded", "log_time.csv")
    cost = tmw_network_simplex(x, y, bandwidth=w)
    print("Wasserstein distance:", cost)

if __name__ == "__main__":
    main()

