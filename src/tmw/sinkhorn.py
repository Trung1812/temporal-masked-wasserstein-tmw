# entropic OT solver (ε fixed to 1e-7)
import numpy as np
import ot
import warnings
from ot.backend import get_backend
from ot.utils import list_to_array


def get_mask(m, n, w):
    """Get the mask for the locality constraint"""
    
    mask = np.zeros((m, n), dtype=np.float64)
    row_idx, col_idx = np.ogrid[:m, :n]
    mask = np.abs(row_idx - col_idx) <= w
    return mask

def tmw_sinkhorn_knopp(a,
                 b,
                 M,
                 mask,
                 reg,
                 numItermax=1000,
                 stopThr=1e-7,
                 log=False,
                 verbose=False,
                 warn=True,
                 warmstart=None,
                 **kwargs
                 ):
    """Compute the TMW distance between two distributions using Sinkhorn's algorithm.
    Paremters:
    --------
    Same as in ot.bregman.sinkhorn with addtion of `w` parameter
    w : float
        Bandwidth parameter for locality constraint.
    Returns:
    -------
    distance : float
        The TMW distance.
    transport_plan : ndarray"""
    a, b, M, mask = list_to_array(a, b, M, mask)
    nx = get_backend(a, b, M, mask)

    if len(a) == 0:
        a = nx.full((M.shape[0],), 1.0 / M.shape[0], type_as=M)
    if len(b) == 0:
        b = nx.full((M.shape[1],), 1.0 / M.shape[1], type_as=M)

    # init data
    dim_a = len(a)
    dim_b = b.shape[0]

    if len(b.shape) > 1:
        n_hists = b.shape[1]
    else:
        n_hists = 0

    if log:
        log = {"err": []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if warmstart is None:
        if n_hists:
            u = nx.ones((dim_a, n_hists), type_as=M) / dim_a
            v = nx.ones((dim_b, n_hists), type_as=M) / dim_b
        else:
            u = nx.ones(dim_a, type_as=M) / dim_a
            v = nx.ones(dim_b, type_as=M) / dim_b
    else:
        u, v = nx.exp(warmstart[0]), nx.exp(warmstart[1])

    #
    K = mask * nx.exp(M / (-reg))

    Kp = (1 / a).reshape(-1, 1) * K

    err = 1
    for ii in range(numItermax):
        uprev = u
        vprev = v
        KtransposeU = nx.dot(K.T, u)
        v = b / KtransposeU
        u = 1.0 / nx.dot(Kp, v)

        if (
            nx.any(KtransposeU == 0)
            or nx.any(nx.isnan(u))
            or nx.any(nx.isnan(v))
            or nx.any(nx.isinf(u))
            or nx.any(nx.isinf(v))
        ):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn("Warning: numerical errors at iteration %d" % ii)
            u = uprev
            v = vprev
            break
        if ii % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            if n_hists:
                tmp2 = nx.einsum("ik,ij,jk->jk", u, K, v)
            else:
                # compute right marginal tmp2= (diag(u)Kdiag(v))^T1
                tmp2 = nx.einsum("i,ij,j->j", u, K, v)
            err = nx.norm(tmp2 - b)  # violation of marginal
            if log:
                log["err"].append(err)

            if err < stopThr:
                break
            if verbose:
                if ii % 200 == 0:
                    print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
                print("{:5d}|{:8e}|".format(ii, err))
    else:
        if warn:
            warnings.warn(
                "Sinkhorn did not converge. You might want to "
                "increase the number of iterations `numItermax` "
                "or the regularization parameter `reg`."
            )
    if log:
        log["niter"] = ii
        log["u"] = u
        log["v"] = v

    if n_hists:  # return only loss
        res = nx.einsum("ik,ij,jk,ij->k", u, K, v, M)
        if log:
            return res, log
        else:
            return res

    else:  # return OT matrix
        if log:
            return u.reshape((-1, 1)) * K * v.reshape((1, -1)), log
        else:
            return u.reshape((-1, 1)) * K * v.reshape((1, -1))


def tmw_sinkhorn_log(
    a,
    b,
    M,
    mask,
    reg,
    numItermax=1000,
    stopThr=1e-7,
    verbose=False,
    log=False,
    warn=True,
    warmstart=None,
    **kwargs,
):
    r"""
    Solve the entropic regularization TMW problem in log space
    and return the OT matrix

    Parameters
    ----------
    Like in ot.bregman.sinkhorn_log with addtion of `mask` parameter
    mask : array-like, shape (dim_a, dim_b)
        Bandwidth parameter for locality constraint.
    Returns
    -------
    gamma : array-like, shape (dim_a, dim_b)
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    Examples
    --------

    >>> import ot
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> M=[[0., 1.], [1., 0.]]
    >>> ot.sinkhorn(a, b, M, 1)
    array([[0.36552929, 0.13447071],
           [0.13447071, 0.36552929]])


    .. _references-sinkhorn-log:
    References
    ----------

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of
        Optimal Transport, Advances in Neural Information Processing
        Systems (NIPS) 26, 2013

    .. [34] Feydy, J., Séjourné, T., Vialard, F. X., Amari, S. I.,
        Trouvé, A., & Peyré, G. (2019, April). Interpolating between
        optimal transport and MMD using Sinkhorn divergences. In The
        22nd International Conference on Artificial Intelligence and
        Statistics (pp. 2681-2690). PMLR.


    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT

    """

    a, b, M, mask = list_to_array(a, b, M, mask)

    nx = get_backend(M, a, b, mask)
    
    if len(a) == 0:
        a = nx.full((M.shape[0],), 1.0 / M.shape[0], type_as=M)
    if len(b) == 0:
        b = nx.full((M.shape[1],), 1.0 / M.shape[1], type_as=M)

    # init data
    dim_a = len(a)
    dim_b = b.shape[0]

    if len(b.shape) > 1:
        n_hists = b.shape[1]
    else:
        n_hists = 0

    # in case of multiple histograms
    if n_hists > 1 and warmstart is None:
        warmstart = [None] * n_hists

    if n_hists:  # we do not want to use tensors sor we do a loop
        lst_loss = []
        lst_u = []
        lst_v = []

        for k in range(n_hists):
            res = tmw_sinkhorn_log(
                a,
                b[:, k],
                M,
                mask,
                reg,
                numItermax=numItermax,
                stopThr=stopThr,
                verbose=verbose,
                log=log,
                warmstart=warmstart[k],
                **kwargs,
            )

            if log:
                lst_loss.append(nx.sum(M * res[0]))
                lst_u.append(res[1]["log_u"])
                lst_v.append(res[1]["log_v"])
            else:
                lst_loss.append(nx.sum(M * res))
        res = nx.stack(lst_loss)
        if log:
            log = {
                "log_u": nx.stack(lst_u, 1),
                "log_v": nx.stack(lst_v, 1),
            }
            log["u"] = nx.exp(log["log_u"])
            log["v"] = nx.exp(log["log_v"])
            return res, log
        else:
            return res

    else:
        if log:
            log = {"err": []}

        def get_logT(u, v):
            if n_hists:
                return Mr[:, :, None] + u + v
            else:
                return Mr + u[:, None] + v[None, :]
        Mr = -M / reg + nx.log(mask)

        # we assume that no distances are null except those of the diagonal of
        # distances
        if warmstart is None:
            u = nx.zeros(dim_a, type_as=M)
            v = nx.zeros(dim_b, type_as=M)
        else:
            u, v = warmstart

        loga = nx.log(a)
        logb = nx.log(b)

        err = 1
        for ii in range(numItermax):
            v = logb - nx.logsumexp(Mr + u[:, None], 0)
            u = loga - nx.logsumexp(Mr + v[None, :], 1)

            if ii % 10 == 0:
                # we can speed up the process by checking for the error only all
                # the 10th iterations

                # compute right marginal tmp2= (diag(u)Kdiag(v))^T1
                tmp2 = nx.sum(nx.exp(get_logT(u, v)), 0)
                err = nx.norm(tmp2 - b)  # violation of marginal
                if log:
                    log["err"].append(err)

                if verbose:
                    if ii % 200 == 0:
                        print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
                    print("{:5d}|{:8e}|".format(ii, err))
                if err < stopThr:
                    break
        else:
            if warn:
                warnings.warn(
                    "Sinkhorn did not converge. You might want to "
                    "increase the number of iterations `numItermax` "
                    "or the regularization parameter `reg`."
                )

        if log:
            log["niter"] = ii
            log["log_u"] = u
            log["log_v"] = v
            log["u"] = nx.exp(u)
            log["v"] = nx.exp(v)

            return nx.exp(get_logT(u, v)), log

        else:
            return nx.exp(get_logT(u, v))

import warnings

def tmw_sinkhorn_knopp_batch(a,
                             b,
                             M,
                             mask,
                             reg,
                             numItermax=1000,
                             stopThr=1e-7,
                             log=False,
                             verbose=False,
                             warn=True,
                             warmstart=None,
                             **kwargs
                            ):
    """
    Batch Sinkhorn for a single (a,b) but multiple cost matrices.
    
    a           : array-like, shape (dim_a,)
    b           : array-like, shape (dim_b,)
    M_list      : array-like, shape (batch, dim_a, dim_b)
    mask        : array-like, shape (dim_a, dim_b) or (batch, dim_a, dim_b)
    reg         : float, regularization
    numItermax, stopThr, log, verbose, warn, warmstart : same meaning as tmw_sinkhorn_knopp

    Returns
    -------
    plans      : ndarray, shape (batch, dim_a, dim_b)
    (and logs if log=True)
    """
    # ensure arrays
    a, b, M, mask = list_to_array(a, b, M, mask)
    
    batch, dim_a, dim_b = M.shape
  
    nx = get_backend(a, b, M, mask)

    # init u, v for each batch
    if warmstart is not None:
        u, v = nx.exp(warmstart[0]), nx.exp(warmstart[1])
    else:
        u = nx.ones((batch, dim_a), type_as=M) / dim_a
        v = nx.ones((batch, dim_b), type_as=M) / dim_b

    # build kernel once per batch
    K  = mask * nx.exp(M / (-reg))
    Kp = (1.0 / a).reshape(1, -1, 1) * K       # a shape -> (1, dim_a, 1) K shape = M shape (batch, dim_a, dim_b)

    if log:
        err_log = []

    # Sinkhorn loop, vectorized
    for ii in range(numItermax):
        up, vp = u, v

        # update v: K^T u
        Kt_u = nx.einsum('bij,bi->bj', K, u)    # shape (batch, dim_b)
        v    = b.reshape(1, -1) / Kt_u           # broadcast b over batch

        # update u: Kp v
        u    = 1.0 / nx.einsum('bij,bj->bi', Kp, v)

        # check numeric issues
        if (nx.any(Kt_u == 0) or nx.any(nx.isnan(u)) or nx.any(nx.isnan(v))
            or nx.any(nx.isinf(u)) or nx.any(nx.isinf(v))):
            warnings.warn(f"[batch Sinkhorn] numerical error at iter {ii}")
            u, v = up, vp
            break

        # convergence check every 10 iters
        if ii % 10 == 0:
            # current b‐marginal: sum_i u*K*v
            tmp2 = nx.einsum('bi,bij,bj->bj', u, K, v)
            err  = nx.norm(tmp2 - b.reshape(1, -1), axis=1)  # per batch
            if log:
                err_log.append(err)
            if (err < stopThr).all():
                break
            if verbose and ii % 200 == 0:
                print("It.| Err\n" + "-"*19)
            if verbose:
                for bi, e in enumerate(err):
                    print(f"[batch {bi}] {ii:5d} | {e:.2e}")

    else:
        if warn:
            warnings.warn(
                "[batch Sinkhorn] did not converge; "
                "consider increasing numItermax or reg."
            )

    # collect logs
    if log:
        log_dict = {'err': nx.stack(err_log, axis=0),
                    'niter': ii,
                    'u': u,
                    'v': v}

    # final transport plans
    plans = u[:, :, None] * K * v[:, None, :]

    return (plans, log_dict) if log else plans

def tmw_sinkhorn2(
    a,
    b,
    M,
    mask,
    reg,
    method="sinkhorn",
    numItermax=1000,
    stopThr=1e-7,
    verbose=False,
    log=False,
    warn=False,
    warmstart=None,
    **kwargs,
):
    r"""
    Solve the entropic regularization TMW problem and return the loss

    **Choosing a Sinkhorn solver**

    By default and when using a regularization parameter that is not too small
    the default sinkhorn solver should be enough. If you need to use a small
    regularization to get sharper OT matrices, you should use the
    :py:func:`ot.bregman.sinkhorn_log` solver that will avoid numerical
    errors.

    Parameters
    ----------
    same as in :py:func:`ot.bregman.sinkhorn2` with addtion of `mask` parameter
    mask:   array-like, shape (dim_a, dim_b)
        Bandwidth parameter for locality constraint.
        If the mask is not provided, it is assumed that all distances are valid
        (i.e. no locality constraint).
    -------
    W : (n_hists) float/array-like
        Optimal transportation loss for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    """

    M, a, b, mask = list_to_array(M, a, b, mask)
    nx = get_backend(M, a, b, mask)

    if len(b.shape) < 2:
        if method.lower() == "sinkhorn":
            res = tmw_sinkhorn_knopp(
                a,
                b,
                M,
                mask,
                reg,
                numItermax=numItermax,
                stopThr=stopThr,
                verbose=verbose,
                log=log,
                warn=warn,
                warmstart=warmstart,
                **kwargs,
            )
        elif method.lower() == "sinkhorn_log":
            res = tmw_sinkhorn_log(
                a,
                b,
                M,
                mask,
                reg,
                numItermax=numItermax,
                stopThr=stopThr,
                verbose=verbose,
                log=log,
                warn=warn,
                warmstart=warmstart,
                **kwargs,
            )
        else:
            raise ValueError("Unknown method '%s'." % method)
        if log:
            return nx.sum(M * res[0]), res[1]
        else:
            return nx.sum(M * res)

    else:
        if method.lower() == "sinkhorn":
            return tmw_sinkhorn_knopp(
                a,
                b,
                M,
                mask,
                reg,
                numItermax=numItermax,
                stopThr=stopThr,
                verbose=verbose,
                log=log,
                warn=warn,
                warmstart=warmstart,
                **kwargs,
            )
        elif method.lower() == "sinkhorn_log":
            return tmw_sinkhorn_log(
                a,
                b,
                M,
                reg,
                numItermax=numItermax,
                stopThr=stopThr,
                verbose=verbose,
                log=log,
                warn=warn,
                warmstart=warmstart,
                **kwargs,
            )
        else:
            raise ValueError("Unknown method '%s'." % method)
  

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    a = np.ones(5) / 5
    b = np.ones(5) / 5
    M = np.random.rand(5, 5)
    mask = get_mask(5, 5, 2)
    reg = 0.1

    distance = tmw_sinkhorn2(a, b, M, mask, reg, method="sinkhorn", numItermax=1000, stopThr=1e-7)
    print("TMW Distance:", distance)

    # Example usage with log
    distance_log = tmw_sinkhorn2(a, b, M, mask, reg, method="sinkhorn_log", numItermax=1000, stopThr=1e-7)
    print("TMW Distance (log):", distance_log)


    