"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable, Iterator, List, TypeVar

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# f(x) = 1.0 / (1.0 + e^{-x}) if x >= 0 else e^{x} / (1.0 + e^{x})
# For is_close:
# f(x) = |x - y| < 1e-2

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


def mul(x: float, y: float) -> float:
    """Return the product x * y."""
    return x * y


def id(x: T) -> T:
    """Return the input unchanged."""
    return x


def add(x: float, y: float) -> float:
    """Return the sum x + y."""
    return x + y


def neg(x: float) -> float:
    """Return the additive inverse of x (i.e., -x)."""
    return -x


def lt(x: float, y: float) -> bool:
    """Return True if x < y, else False."""
    return x < y


def eq(x: float, y: float) -> bool:
    """Return True if x == y, else False."""
    return x == y


def max(x: float, y: float) -> float:
    """Return the greater of x and y."""
    return __import__("builtins").max(x, y)


def is_close(x: float, y: float) -> bool:
    """Return True if |x - y| < 1e-2, else False."""
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Return numerically stable logistic sigmoid of x."""
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def relu(x: float) -> float:
    """Return ReLU(x) = max(x, 0)."""
    return max(x, 0)


def log(x: float) -> float:
    """Return natural logarithm of x. Requires x > 0."""
    return math.log(x)


def exp(x: float) -> float:
    """Return e**x."""
    return math.exp(x)


def log_back(x: float, y: float) -> float:
    """Return backward pass for log: d/dx[log(x)] * y = y / x. Requires x != 0."""
    return y / x


def inv(x: float) -> float:
    """Return multiplicative inverse of x (1 / x). Requires x != 0."""
    return 1.0 / x


def inv_back(x: float, y: float) -> float:
    """Return backward pass for inv: d/dx[1/x] * y = -y / x^2. Requires x != 0."""
    return -y / (x * x)


def relu_back(x: float, y: float) -> float:
    """Return backward pass for ReLU: y if x > 0 else 0."""
    return y if x > 0 else 0.0


# ## Task 0.3
#
# Small practice library of elementary higher-order functions.
#
# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(a: Iterable[T], f: Callable[[T], U]) -> List[U]:
    """Apply unary function f to each element of iterable a and return a list of results."""
    return [f(x) for x in a]


def zipWith(a: Iterable[T], b: Iterable[U], f: Callable[[T, U], V]) -> Iterator[V]:
    """Combine two iterables elementwise with binary function f, yielding results until either is exhausted."""
    it_a, it_b = iter(a), iter(b)
    while True:
        try:
            x, y = next(it_a), next(it_b)
        except StopIteration:
            return
        yield f(x, y)


def reduce(
    a: Iterable[float], f: Callable[[float, float], float], s: float = 0.0
) -> float:
    """Fold iterable a from the left with binary function f starting at seed s."""
    acc = s
    for it in a:
        acc = f(acc, it)
    return acc


def negList(a: Iterable[float]) -> List[float]:
    """Return a list with each element of a negated."""
    return map(a, lambda x: -x)


def addLists(a: Iterable[float], b: Iterable[float]) -> List[float]:
    """Return a list containing elementwise sums of iterables a and b (up to the shorter length)."""
    res: List[float] = []
    for i in zipWith(a, b, lambda x, y: x + y):
        res.append(i)
    return res


def sum(a: Iterable[float]) -> float:
    """Return the arithmetic sum of elements in a."""
    return reduce(a, lambda x, y: x + y, 0.0)


def prod(a: Iterable[float]) -> float:
    """Return the product of elements in a."""
    return reduce(a, lambda x, y: x * y, 1.0)
