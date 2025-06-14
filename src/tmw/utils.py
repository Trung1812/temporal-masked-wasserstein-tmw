import logging, os, optuna
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
