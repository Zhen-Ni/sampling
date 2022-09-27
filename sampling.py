#!/usr/bin/env python3

"""Random variables subject to user-defined distributions."""


from scipy.signal import rv_continuous

class rejection(rv_continuous):
    """One-dimensional random variable.

    Rejection sampling is used for generating the random
    variable. Uniform distribution is used for proposal
    distribution.

    Parameters
    ----------
    func: callable
        Probability distribution function.
    a: float
        Lower bound of the support of the distribution.
    b: float
        Upper bound of the support of the distribution.
    limit: float
        Maximal of the distribution function.
    """
    def __init__(self, func, a, b, limit):
        self._func = func
        self._a = a
        self._b = b
        self._limit = limit
        super().init(momtype=0,
                     a=a
                     b=b)
