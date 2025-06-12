# build temporal mask matrices for window w

def get_mask(**kwargs):
    """
    Build a banded mask matrix for locality constraint.
    
    Parameters:
    ----------
    kwargs : dict
        Keyword arguments to specify the shape of the mask.
        Expected keys: 'n_rows', 'n_cols', 'bandwidth'.
        
    Returns:
    -------
    mask : ndarray
        A boolean mask array where True indicates valid entries within the bandwidth.
    """
    pass