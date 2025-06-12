# z-normalization, loading, splitting
import numpy as np
import pandas as pd
def load_dataset(file_path):
    """
    Load dataset from a file.
    Args:
        file_path (str): Path to the dataset file.
    Returns:
        X (numpy.ndarray): Feature matrix of shape (n_samples, n_features).
        y (numpy.ndarray): Labels of shape (n_samples,).
    """
    import numpy as np
    data = pd.read_csv(file_path)
    # Assuming the last column is the label and the rest are features
    if 'y' in data.columns:
        X = data.drop(columns=['y']).values
        y = data['y'].values
    else:
        X = data.values[:, :-1]
        y = data.values[:, -1]
    X = data['X']
    y = data['y']
    return X, y