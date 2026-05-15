"""Metrics utilities for basic statistical calculations."""

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