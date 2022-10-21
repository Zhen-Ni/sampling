#!/usr/bin/env python3

"""Random variables subject to user-defined distributions."""

import math
import random

from typing import Callable, Concatenate, ParamSpec, TypeVar
from collections.abc import Sequence, Iterable


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
    pdf : callable
        Probability distribution function.
    a : float
        Lower bound of the support of the distribution.
    b : float
        Upper bound of the support of the distribution.
    limit : float
        Maximal of the distribution function.
    """
    while True:
        res = random.random()
        res *= b - a
        res += a
        if pdf(res) > limit * random.random():
            break
    return res


P = ParamSpec('P')
T = TypeVar('T', bound=float | Sequence[float])


def metropolis(pdf: Callable[Concatenate[T, P], float],
               size: int,
               initial: T,
               std: float | T = 1.
               ) -> list[T]:
    """A sequence of Multi-dimensional random variables.

    The metropolis algorithm is used for generating
    multi-dimensional random variables. The random variables
    may have random walk behavior.

    Gaussian distribution is chosen to be the proposal
    distribution.

    Parameters
    ----------
    pdf : callable
        Probability distribution function.
    size : int
        Number of random variables to generate.
    initial: float or sequence of float
        Initial value for the Markov chain.
    std : float or sequence of float, optional
        The standard deviation of the gaussian distribution.
        If it is a sequence of floats, it should has the same
        size with dimension of `pdf` function. Defaults
        to 1.
    """
    if isinstance(initial, Sequence):
        chain = _metropolis_iterable(pdf, size, initial, std)
    else:
        chain = _metropolis_scalar(pdf, size, initial, std)
    return chain


def _metropolis_scalar(pdf: Callable[Concatenate[float, P], float],
                       size: int,
                       initial: float,
                       std: float = 1
                       ) -> list[float]:
    z_old = initial
    chain = [z_old]
    for i in range(size):
        z_new = gaussian() * std + z_old
        if min(1, pdf(z_new) / pdf(z_old)) < random.random():
            z_new = z_old
        chain.append(z_new)
        z_old = z_new
    return chain


def _metropolis_iterable(pdf: Callable[Concatenate[Sequence[float],
                                                   P], float],
                         size: int,
                         initial: Sequence[float],
                         std: float | Sequence[float] = 1
                         ) -> list[tuple[float, ...]]:
    z_old = tuple(initial)
    if not isinstance(std, Iterable):
        std = tuple([std] * len(z_old))
    chain = [z_old]
    for i in range(size - 1):
        z_new = tuple([gaussian() * sigma + mu
                       for sigma, mu in zip(std, z_old)])
        if min(1, pdf(z_new) / pdf(z_old)) < random.random():
            z_new = z_old
        chain.append(z_new)
        z_old = z_new
    return chain


def _test_gaussian():
    samples = [gaussian() for i in range(100000)]
    plt.figure()
    plt.hist(samples, bins=50, density=True)


def _test_rejection():
    def pdf(x): return 1 - x ** 2
    samples = [rejection(pdf, -1, 1, 1) for i in range(100000)]
    plt.figure()
    plt.hist(samples, bins=50, density=True)


def _test_metropolis():
    def pdf(x):
        x0, x1 = x
        return math.exp(-x0 ** 2 - (2*x1) ** 2 - x0 * x1)

    size = 10000
    initial = [0, 0]
    std = 1
    chain = metropolis(pdf, size, initial, std)
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot(*zip(*chain), 'o-', lw=0.1, ms=0.5, alpha=0.5)


if __name__ == '__main__':
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl_fontpath = mpl.get_data_path() + '/fonts/ttf/STIXGeneral.ttf'
    mpl_fontprop = mpl.font_manager.FontProperties(fname=mpl_fontpath)
    plt.rc('font', family='STIXGeneral', weight='normal', size=10)
    plt.rc('mathtext', fontset='stix')
    _test_gaussian()
    _test_rejection()
    _test_metropolis()
    plt.show()
