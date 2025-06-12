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
    data = pd.read_csv(file_path, sep='\t', header=None)
    # Assuming the last column is the label and the rest are features
    
    X = data.values[:, 1:]
    y = data.values[:, 0]
    
    return X, y

def z_normalize(X):
    """
    Perform z-normalization on the feature matrix.
    Args:
        X (numpy.ndarray): Feature matrix of shape (n_samples, n_features).
    Returns:
        X_normalized (numpy.ndarray): Z-normalized feature matrix.
    """
    #NOTE: I should use different normalization for time series datasets, datasets in 
    # UCR are already normalized
    pass

if __name__ == "__main__":
    # Example usage
    file_path = "data/processed/BeetleFly/BeetleFly_TRAIN.tsv"
    X, y = load_dataset(file_path)
    print("Feature matrix shape:", X.shape)
    print("Labels shape:", y.shape)
    
    # Perform z-normalization
    X_normalized = z_normalize(X)
    print("Normalized feature matrix shape:", X_normalized.shape)