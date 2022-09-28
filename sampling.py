#!/usr/bin/env python3

"""Random variables subject to user-defined distributions."""

import math
import random

from typing import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl_fontpath = mpl.get_data_path() + '/fonts/ttf/STIXGeneral.ttf'
mpl_fontprop = mpl.font_manager.FontProperties(fname=mpl_fontpath)
plt.rc('font', family='STIXGeneral', weight='normal', size=10)
plt.rc('mathtext', fontset='stix')


def gaussian() -> float:
    """Samples from standard Gaussian distribution.

    The Box-Muller method is used for generating the
    random variables. The Box-Muller method can
    generate two gaussian random variables at a time.
    But in this function, only one random variable
    will be calculated and returned.
    """
    r2 = 2.
    while 1 < r2:
        z1 = random.random() * 2 - 1
        z2 = random.random() * 2 - 1
        r2 = z1 ** 2 + z2 ** 2
    return z1 * (-math.log(r2) / r2) ** .5


def rejection(pdf: Callable[[float], float],
              a: float,
              b: float,
              limit: float,
              ) -> float:
    """One-dimensional random variable.

    Rejection sampling is used for generating the random
    variable. Uniform distribution is used for proposal
    distribution.

    Parameters
    ----------
    pdf: callable
        Probability distribution function.
    a: float
        Lower bound of the support of the distribution.
    b: float
        Upper bound of the support of the distribution.
    limit: float
        Maximal of the distribution function.
    size: int or tuple of ints, optional
        Output shape. Default is None, in which case a
        single value is returned.
    """
    while True:
        res = random.random()
        res *= b - a
        res += a
        if pdf(res) > limit * random.random():
            break
    return res


def _test_gaussian():
    samples = [gaussian() for i in range(100000)]
    plt.figure()
    plt.hist(samples, bins=50, density=True)


def _test_rejection():
    pdf = lambda x: 1 - x ** 2
    samples = [rejection(pdf, -1, 1, 1) for i in range(100000)]
    plt.figure()
    plt.hist(samples, bins=50, density=True)


if __name__ == '__main__':
    _test_gaussian()
    _test_rejection()
    plt.show()
