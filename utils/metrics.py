"""Metrics utilities for basic statistical calculations."""
import numpy as np
import math
from numpy.typing import NDArray

def mean(data):
    """Return the arithmetic mean of a list of numbers."""
    if len(data) == 0:
        return 0
    return sum(data) / len(data)


def variance(data, ddof=1):
    """Return the variance of a list of numbers."""
    n = len(data)
    if n <= ddof:
        return 0
    m = mean(data)
    return sum((x - m) ** 2 for x in data) / (n - ddof)


def std_dev(data, ddof=1):
    """Return the standard deviation of a list of numbers."""
    return variance(data, ddof=ddof) ** 0.5


def joint_probability(p_a, p_b_given_a):
    """Return P(A and B) = P(A) * P(B|A)."""
    return p_a * p_b_given_a


def conditional_probability(joint, p_a):
    """Return P(B|A) = P(A and B) / P(A)."""
    if p_a == 0:
        return 0
    return joint / p_a


def bayes_theorem(p_b_given_a, p_a, p_b):
    """Return P(A|B) using Bayes' theorem."""
    if p_b == 0:
        return 0
    return (p_b_given_a * p_a) / p_b

def accuracy(y_true: NDArray, y_pred: NDArray) -> float:
    return np.mean(y_true == y_pred)

def precision(y_true: NDArray, y_pred: NDArray) -> float:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) != 0 else 0.0

def recall(y_true: NDArray, y_pred: NDArray) -> float:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) != 0 else 0.0