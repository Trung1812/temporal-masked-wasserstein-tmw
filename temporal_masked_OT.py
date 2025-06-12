from . import utils,linearprog,sinkhorn,KPG_GW,partial_OT
import numpy as np
import torch
import optim
import abc

class TemporalMaskedWasserstein(object):
    def __init__(self):
        super().__init__()

    def cost_matrix(self,xs,xt,cost_function="L2",eps=1e-10):
        if cost_function == "L2":
            return utils.cost_matrix(xs,xt)
        elif cost_function == "cosine":
            xs = xs / (np.linalg.norm(xs, axis=1, keepdim=True) + eps)
            xt = xt / (np.linalg.norm(xt, axis=1, keepdim=True) + eps)
            return 0.5*utils.cost_matrix(xs, xt)
        else:
            return cost_function(xs,xt)
    
    @abc.classmethod
    def mask_matrix(self,xs,xt,mask_type,kappa,tau,eps_threshold):
        ## mask matrix type 1
        if mask_type==1:
            fx = np.arange(len(xs))
            fy = np.arange(len(xt))
        ## mask matrix type 2
        elif mask_type==2:
            xs = np.array(xs).reshape(-1, 1)
            xt = np.array(xt).reshape(-1, 1)
            Cs:np.ndarray = np.linalg.norm(xs[:-1] - xs[1:], axis=1)
            Ct:np.ndarray = np.linalg.norm(xt[:-1] - xt[1:], axis=1)
            fx = np.concatenate(([0], np.cumsum(Cs)))
            fy = np.concatenate(([0], np.cumsum(Ct)))
            fx /= fx[-1]
            fy /= fy[-1]
        if mask_type!=3:
            ## calculate the mask matrix
            diff_matrix = np.abs(fx-fy.reshape(-1,1))
            sigmoid_matrix = 1 / (1 + np.exp(-kappa * (diff_matrix - tau))) 
            M = (sigmoid_matrix < eps_threshold).astype(int)
            return M
        else:
            diff_matrix_1 = self.mask_matrix(xs,xt,1,kappa,tau,eps_threshold)
            diff_matrix_2 = self.mask_matrix(xs,xt,2,kappa,tau,eps_threshold)
            return 0.5*diff_matrix_1+0.5*diff_matrix_2

    def tmw(self,p,q,xs,xt,cost_function="L2",mask_type=1,algorithm="linear_programming",normalized=True,
               reg=0.0001,max_iterations=100000,thres=1e-5,eps=1e-10,eps_threshold=0.01,kappa=0.01,tau=0.01,masked=True):
        '''
        :param p: ndarray, (m,), Mass of source samples
        :param q: ndarray, (n,), Mass of target samples
        :param xs: ndarray, (m,d), d-dimensional source samples
        :param xt: ndarray, (n,d), d-dimensional target samples
        :param K: list of tuples, e.g., [(0,1),(10,20)]. Each tuple is an index pair of keypoints.
        :param cost_function: str or function, type of cost function. Default is "L2". Choices should be "L2", "cosine",
        and a pre-defined function.
        :param mask_type: type of masking to use with the distance. Default is 1.
        :param algorithm: str, algorithm to solve model. Default is "linear_programming". Choices should be
        "linear_programming" and "sinkhorn".
        :param tau_s: float, source temperature for computing the relation.
        :param tau_t: float, target temperature for computing the relation.
        :param normalized: bool, whether to normalize the distance
        :param reg: float, regularization coefficient in entropic model
        :param max_iterations: int, maximum number of iterations
        :param eps: float, a small number to avoid NaN
        :param thres: float, stop criterion for sinkhorn
        :return: transport plan, (m,n)
        :eps_threshold: #todo
        :kappa: #todo
        :tau: #todo
        '''
        ## Cost matrix
        C = self.cost_matrix(xs,xt,cost_function,eps)
        if normalized:
            C /= (C.max() + 0)
        if masked:
            M = self.mask_matrix(xs,xt,mask_type,kappa,tau,eps_threshold)
        else:
            M = np.ones([xs.shape[0],xt.shape[0]])
        ## solving model
        if algorithm == "linear_programming":
            pi = linearprog.lp(p,q,C,M)
        elif algorithm == "sinkhorn":
            pi = sinkhorn.sinkhorn_log_domain(p,q,C,M,reg,max_iterations,thres)
        else:
            raise ValueError("algorithm must be 'linear_programming' or 'sinkhorn'!")
        return pi

    def tmwd(self,p,q,xs,xt,cost_function="L2",mask_type=1,algorithm="linear_programming",normalized=True,
               reg=0.0001,max_iterations=100000,thres=1e-5,eps=1e-10,eps_threshold=0.01,kappa=0.01,tau=0.01,masked=True):
        '''
        :param p: ndarray, (m,), Mass of source samples
        :param q: ndarray, (n,), Mass of target samples
        :param xs: ndarray, (m,d), d-dimensional source samples
        :param xt: ndarray, (n,d), d-dimensional target samples
        :param K: list of tuples, e.g., [(0,1),(10,20)]. Each tuple is an index pair of keypoints.
        :param cost_function: str or function, type of cost function. Default is "L2". Choices should be "L2", "cosine",
        and a pre-defined function.
        :param mask_type: type of masking to use with the distance. Default is 1.
        :param algorithm: str, algorithm to solve model. Default is "linear_programming". Choices should be
        "linear_programming" and "sinkhorn".
        :param tau_s: float, source temperature for computing the relation.
        :param tau_t: float, target temperature for computing the relation.
        :param normalized: bool, whether to normalize the distance
        :param reg: float, regularization coefficient in entropic model
        :param max_iterations: int, maximum number of iterations
        :param eps: float, a small number to avoid NaN
        :param thres: float, stop criterion for sinkhorn
        :return: transport plan, (m,n)
        :eps_threshold: #todo
        :kappa: #todo
        :tau: #todo
        '''
        ## Cost matrix
        C = self.cost_matrix(xs,xt,cost_function,eps)
        if normalized:
            C /= (C.max() + 0)
        if masked:
            M = self.mask_matrix(xs,xt,mask_type,kappa,tau,eps_threshold)
        else:
            M = np.ones([xs.shape[0],xt.shape[0]])
        ## solving model
        if algorithm == "linear_programming":
            pi = linearprog.lp(p,q,C,M)
        elif algorithm == "sinkhorn":
            pi = sinkhorn.sinkhorn_log_domain(p,q,C,M,reg,max_iterations,thres)
        else:
            raise ValueError("algorithm must be 'linear_programming' or 'sinkhorn'!")
        return np.sum(np.multiply(np.multiply(pi,M),C))

tmw = TemporalMaskedWasserstein()
def tmwdist(u_values, v_values, my_p, q, cost_function="L2", mask_type=1, algorithm="linear_programming",
            normalized=True, reg=0.0001, max_iterations=100000, thres=1e-5, eps=1e-10, eps_threshold=0.01,
            kappa=0.01, tau=0.01, masked=True):
    
    return tmw.tmwd(my_p, q, u_values, v_values, cost_function, mask_type, algorithm,
                    normalized, reg, max_iterations, thres, eps, eps_threshold, kappa, tau, masked)
if __name__ == "__main__":
    
