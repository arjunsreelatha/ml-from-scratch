def mean(data):
    return sum(data) / (len(data))
# use population or sample denominator later(n or n-1?)
def variance(data):
    m = mean(data)
    return sum((x - m) ** 2 for x in data) / (len(data))    
# square root of variance
def std_dev(data):
    return variance(data) ** 0.5
