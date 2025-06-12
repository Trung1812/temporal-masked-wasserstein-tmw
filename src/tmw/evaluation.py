import numpy as np
import torch
import optuna
import ot
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, LeaveOneOut

# Import your TMW pipeline components
from src.tmw.preprocessing import load_dataset, z_normalize
from src.tmw.mask import build_mask_matrix
from src.tmw.sinkhorn import tmw_sinkhorn

# Setup device for GPU acceleration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_tmw_distance_matrix(X, w, lam):
    """
    Compute the pairwise TMW distance matrix using POT with torch on GPU if available.
    X: numpy array of shape (n_samples, series_length)
    w: temporal window width
    lam: entropic regularization coefficient
    """
    X_tensor = torch.from_numpy(X).float().to(device)
    n, T = X_tensor.shape

    # Build mask matrix and move to device
    mask = build_mask_matrix(T, w)           # returns numpy array of 0/1
    mask_tensor = torch.from_numpy(mask).float().to(device)

    # Uniform distributions for source/target
    a = torch.ones((T,), device=device) / T
    b = a.clone()

    # Distance matrix placeholder
    D = torch.zeros((n, n), device=device)

    # Compute pairwise OT distances
    for i in range(n):
        xi = X_tensor[i].unsqueeze(0)  # shape (1, T)
        for j in range(i + 1, n):
            xj = X_tensor[j].unsqueeze(0)

            # Squared Euclidean cost matrix
            M = (xi.T - xj).pow(2)  # shape (T, T)
            # Apply temporal mask: high cost where mask==0
            M = M + (1.0 - mask_tensor) * 1e6

            # Compute regularized OT cost (squared distance)
            cost_sq = ot.sinkhorn2(a, b, M, lam,
                                  method='sinkhorn', numItermax=200,
                                  stopThr=1e-9)
            dist = torch.sqrt(cost_sq)

            D[i, j] = dist
            D[j, i] = dist

    return D.cpu().numpy()


def objective(trial):
    # Access global data
    X = OBJ_DATA['X']
    y = OBJ_DATA['y']

    # Suggest hyperparameters
    w = trial.suggest_int('w', 1, X.shape[1] // 2)
    lam = trial.suggest_loguniform('lambda', 1e-3, 10.0)

    # Compute the distance matrix (uses GPU if available)
    D = compute_tmw_distance_matrix(X, w, lam)

    # 1-NN classifier with precomputed distances
    knn = KNeighborsClassifier(n_neighbors=1, metric='precomputed')
    cv = LeaveOneOut()
    scores = cross_val_score(knn, D, y, cv=cv,
                             scoring='accuracy', n_jobs=-1)
    return scores.mean()


if __name__ == '__main__':
    # Load and preprocess data
    X_raw, y = load_dataset('data/processed/train.csv')
    X = z_normalize(X_raw)

    # Store data for objective
    OBJ_DATA = {'X': X, 'y': y}

    # Create Optuna study and optimize
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    # Report results
    print("Best parameters:", study.best_params)
    print("Best LOOCV accuracy:", study.best_value)
