import os
import time
import argparse
import logging

import torch
import optuna
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score

from preprocessing import load_dataset
from sinkhorn import tmw_sinkhorn2, get_mask

N_JOBS = 2
# Setup device for GPU acceleration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_logging(log_dir="logs", log_file="optuna_tuning.log"):
    """
    Creates log directory and file; logs to both file and console.
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, log_file))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)

    # Optuna logging
    optuna.logging.get_logger("optuna").addHandler(ch)
    optuna.logging.set_verbosity(optuna.logging.INFO)


def compute_tmw_distance_matrix(X, w, lam):
    """
    Computes the full pairwise TMW distance matrix on X (train/train).
    """
    X_tensor = torch.from_numpy(X).float().to(device)
    n, T = X_tensor.shape

    mask = get_mask(T, T, w)
    mask_tensor = torch.from_numpy(mask).float().to(device)
    a = torch.ones(T, device=device) / T
    b = a.clone()

    D = torch.zeros((n, n), device=device)
    for i in range(n):
        xi = X_tensor[i].unsqueeze(0).unsqueeze(2)  # Add batch and channel dimensions
        for j in range(i + 1, n):
            xj = X_tensor[j].unsqueeze(0).unsqueeze(2)  # Add batch and channel dimensions
            M = torch.cdist(xi, xj, p=2).squeeze()
            dist = tmw_sinkhorn2(a, b, M, mask_tensor, lam,
                              method='sinkhorn', numItermax=1000,
                              stopThr=1e-7)
            D[i, j] = dist
            D[j, i] = dist

    return D.cpu().numpy()


def compute_tmw_train_test(X_train, X_test, w, lam):
    """
    Computes the test-train TMW distance matrix (n_test * n_train).
    """
    X_tr = torch.from_numpy(X_train).float().to(device)
    X_te = torch.from_numpy(X_test).float().to(device)
    n_tr, T = X_tr.shape
    n_te, _ = X_te.shape

    mask = get_mask(T, T, w)
    mask_tensor = torch.from_numpy(mask).float().to(device)
    a = torch.ones(T, device=device) / T
    b = a.clone()

    D = torch.zeros((n_te, n_tr), device=device)
    for i in range(n_te):
        xi = X_te[i].unsqueeze(0).unsqueeze(2)
        for j in range(n_tr):
            xj = X_tr[j].unsqueeze(0).unsqueeze(2)
            M = torch.cdist(xi, xj, p=2).squeeze()
            dist = tmw_sinkhorn2(a, b, M, mask_tensor, lam,
                              method='sinkhorn', numItermax=1000,
                              stopThr=1e-7)
            D[i, j] = dist

    return D.cpu().numpy()


def objective(trial):
    """
    Optuna objective: LOOCV 1-NN on train with TMW distances.
    """
    X = OBJ_DATA['X']
    y = OBJ_DATA['y']
    w = trial.suggest_int('w', 1, X.shape[1] // 2)
    lam = trial.suggest_loguniform('lambda', 1e-3, 200)
    logging.info(f"Trial {trial.number}: w={w}, lambda={lam:.4f}")

    D = compute_tmw_distance_matrix(X, w, lam)
    knn = KNeighborsClassifier(n_neighbors=1, metric='precomputed')
    scores = cross_val_score(knn, D, y,
                             cv=StratifiedKFold(shuffle=True),
                             scoring='accuracy',
                             n_jobs=N_JOBS)
    acc = scores.mean()
    logging.info(f"Trial {trial.number} accuracy: {acc:.4f}")
    trial.set_user_attr("accuracy", acc)
    return acc


def main():
    parser = argparse.ArgumentParser(
        description="TMW hyperparameter tuning and test inference"
    )
    parser.add_argument(
        "--dataset_dir",
        default="data/processed/SwedishLeaf",
        type=str,
        help="Path to dataset folder containing *_TRAIN.tsv and *_TEST.tsv"
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=5,
        help="Number of Optuna trials"
    )
    parser.add_argument(
        "--n_jobs",
        dest="n_jobs",
        type=int,
        default=2,
        help="Number of parallel jobs for Optuna and cross-validation"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Root directory for logs and outputs"
    )
    args = parser.parse_args()
    global N_JOBS
    N_JOBS = args.n_jobs

    dataset_name = os.path.basename(os.path.normpath(args.dataset_dir))
    out_dir = os.path.join(args.log_dir, dataset_name)
    setup_logging(log_dir=out_dir)
    logging.info(f"Dataset: {dataset_name}")

    train_file = os.path.join(
        args.dataset_dir, f"{dataset_name}_TRAIN.tsv"
    )
    test_file = os.path.join(
        args.dataset_dir, f"{dataset_name}_TEST.tsv"
    )
    if not os.path.isfile(train_file) or not os.path.isfile(test_file):
        logging.error("Train or test file missing in %s", args.dataset_dir)
        return

    logging.info("Loading and normalizing train/test data...")
    X_train, y_train = load_dataset(train_file)
    X_test, y_test = load_dataset(test_file)

    global OBJ_DATA
    OBJ_DATA = {'X': X_train, 'y': y_train}

    storage_url = f"sqlite:///{os.path.join(out_dir, 'optuna.db')}"
    study = optuna.create_study(
        direction='maximize',
        study_name=f"{dataset_name}_tuning",
        storage=storage_url,
        load_if_exists=True
    )
    logging.info("Starting hyperparameter optimization...")
    study.optimize(objective, n_trials=args.n_trials, n_jobs=N_JOBS)

    best = study.best_trial
    best_w = best.params['w']
    best_lam = best.params['lambda']
    logging.info(
        f"Best params (train): w={best_w}, lambda={best_lam:.4f}, "
        f"train_acc={best.value:.4f}"
    )

    # Final train-train distances and fit
    logging.info("Computing final train-train distance matrix…")
    D_train = compute_tmw_distance_matrix(
        X_train, best_w, best_lam
    )
    clf = KNeighborsClassifier(
        n_neighbors=1, metric='precomputed'
    )
    clf.fit(D_train, y_train)

    # Test inference
    logging.info("Computing test-train distances and predicting…")
    start = time.perf_counter()
    D_test = compute_tmw_train_test(
        X_train, X_test, best_w, best_lam
    )
    y_pred = clf.predict(D_test)
    inf_time = time.perf_counter() - start
    test_acc = accuracy_score(y_test, y_pred)

    logging.info(f"Test accuracy: {test_acc:.4f}")
    logging.info(f"Inference time (s): {inf_time:.4f}")

    # Save all trial results
    df = study.trials_dataframe(
        attrs=('number','value','params','user_attrs')
    )
    df.to_csv(
        os.path.join(out_dir, 'optuna_trials.csv'),
        index=False
    )

if __name__ == "__main__":
    main()
