import random


def subsample(cnts, r=1e-5, k=.7):
    summ = sum(cnts)
    freq = [1. * c / summ for c in cnts]
    cnts = [min((r / f) ** k, 1.) * c \
        for c, f in zip(cnts, freq)]
    return cnts

def fussy(weights, f=0.):
    if len(weights) == 0:
        return weights
    if isinstance(weights[0], float):
        f = f * sum(weights)
        weights = [w + f for w in weights]
    elif isinstance(weights[0], list):
        for i in range(len(weights)):
            weights[i] = fussy(weights[i], f)
    else: raise NotImplementedError
    return weights

def sample_idx(weights):
    indices = list(range(len(weights)))
    if max(weights) <= 1e-6: weights=None
    idx = random.choices(indices,
        weights=weights, k=1)[0]
    return idx
