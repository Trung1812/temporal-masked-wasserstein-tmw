# POW benchmark
import numpy as np
import ot

def pow_regularization(M, reg):
    I = get_I(M)
    return M + reg * I


def get_I(M):
    rows, cols = M.shape
    i, j = np.meshgrid(
        np.arange(rows), np.arange(cols), indexing="ij"
    )  # Use np.meshgrid instead of torch.meshgrid
    I = ((i / rows - j / cols) ** 2).astype(M.dtype)
    return I


def get_assignment(soft_assignment):
    """Get assignment from soft assignment"""
    assignment = np.argmax(soft_assignment, axis=0)
    outlier_label = soft_assignment.shape[0] - 1
    assignment[assignment == outlier_label] = -1
    return assignment


def partial_order_wasserstein(
    u_values,
    v_values,
    order_reg,
    m=None,
    p=None,
    q=None,
    nb_dummies=1,
    ot_algo="emd",
    sinkhorn_reg=0.1,
    return_dist=True,
    **kwargs
):
    """Solves the partial optimal transport problem
    and returns the OT plan

    Parameters
    ----------
    M : ndarray, shape (ns, nt)
        Cost matrix
    p : ndarray, shape (ns,), optional
        Masses in the source domain
    q : ndarray, shape (nt,), optional
        Masses in the target domain
    m : float, optional
        Total mass to be transported from source to target
        (default: :math:`\min\{\|\mathbf{p}\|_1, \|\mathbf{q}\|_1\}`)
    order_reg : float
        Order regularization parameter
    nb_dummies : int
        Number of dummy points to add (avoid instabilities in the EMD solver)
    ot_algo : str, optional
        OT solver to use (default: "emd")  either "emd" or "sinkhorn"
    sinkhorn_reg : float, optional
        Sinkhorn regularization parameter (default: 0.1) if ot_algo="sinkhorn"
    return_dist : bool, optional
        If True, returns the partial order wasserstein distance (default: False) else returns the OT plan

    Returns
    -------
    T : ndarray, shape (ns, nt) or float if return_dist=True
    """
    M = ot.dist(np.expand_dims(u_values, axis=1), np.expand_dims(v_values, axis=1))
    if p is None:
        p = np.ones(M.shape[0]) / M.shape[0]
    if q is None:
        q = np.ones(M.shape[1]) / M.shape[1]

    if m is None:
        m = np.min((np.sum(p), np.sum(q)))
    elif m < 0:
        raise ValueError("Problem infeasible. Parameter m should be greater" " than 0.")
    elif m > np.min((np.sum(p), np.sum(q))):
        raise ValueError(
            "Problem infeasible. Parameter m should lower or"
            " equal than min(|a|_1, |b|_1)."
        )

    dim_M_extended = (len(p) + nb_dummies, len(q) + nb_dummies)
    q_extended = np.append(q, [(np.sum(p) - m) / nb_dummies] * nb_dummies)
    p_extended = np.append(p, [(np.sum(q) - m) / nb_dummies] * nb_dummies)

    M_reg = pow_regularization(M, order_reg)

    M_emd = np.zeros(dim_M_extended, dtype=M.dtype)
    M_emd[: len(p), : len(q)] = M_reg
    M_emd[-nb_dummies:, -nb_dummies:] = np.max(M) * 1e2
    if ot_algo == "emd":
        T, logemd = ot.emd(p_extended, q_extended, M_emd, log=True, **kwargs)
    elif ot_algo == "sinkhorn":
        T = ot.sinkhorn(p_extended, q_extended, M_emd, reg=sinkhorn_reg, log=False)

    if return_dist:
        return np.sum(T[: len(p), : len(q)] * M)
    else:
        return T[: len(p), : len(q)]

def main():
    # Example usage
    import timeit
    x = np.random.random_sample((100, ))
    y = np.random.random_sample((100, ))
    print(x.shape, y.shape)
    order_reg = 10
    m = 0.5
    start_time = timeit.default_timer()
    distance = partial_order_wasserstein(x, y, order_reg, m)
    end_time = timeit.default_timer()
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Partial Order Wasserstein Distance: {distance}")
if __name__ == "__main__":
    main()