import numpy as np
from numpy.typing import NDArray


def compute_mse(y_true: NDArray, y_pred: NDArray) -> float:
    """return the mean squared error between true and predicted values"""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    mse = np.mean((y_true - y_pred) ** 2)
    return mse

def bin_cross_entropy(y_true:NDArray,y_pred:NDArray)->np.floating:
    """return the binary cross-entropy loss between true and predicted values"""
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if not np.all((y_true == 0) | (y_true == 1)):
        raise ValueError("y_true must be binary (0 or 1)")
    epsilon = 1e-15
    y_pred = np.clip(y_pred,epsilon,1-epsilon)        
    bce = -(y_true * np.log(y_pred) + (1 - y_true)
            *np.log(1-y_pred))
    return np.mean(bce)
