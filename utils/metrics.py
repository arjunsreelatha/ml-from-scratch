# Metrics utilities for statistical calculations.

#this function provides the mean of a list of numbers
def mean(data):
    if len(data) == 0:
        return 0
    return sum(data) / len(data)
#this function provides the variance of a list of numbers, with an optional degrees of freedom parameter (ddof)(default 1 for sample variance)
def variance(data, ddof=1):
    n = len(data)
    if n <= ddof:
        return 0
    m = mean(data)
    return sum((x - m) ** 2 for x in data) / (n - ddof)
    
#this function provides the standard deviation of a list of numbers, with an optional degrees of freedom parameter (ddof)(default 1 for sample standard deviation
def std_dev(data, ddof=1):
    return variance(data, ddof=ddof) ** 0.5