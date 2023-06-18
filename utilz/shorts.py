"""Shorthands for common functions or `ops` and `maps` with specific params set"""

__all__ = [
    "sort",
    "keep",
    "discard",
    "seq",
    "equal",
    "isempty",
    "checkany",
    "checkall",
    "nth",
    "pairs",
]

from collections.abc import Iterable
from .maps import filter, map
from .ops import curry
from toolz import diff, nth
import itertools as it
import pandas as pd
import numpy as np


@curry
def sort(iterme: Iterable, **kwargs):
    """Alias for `sorted()`"""
    return sorted(iterme, **kwargs)


@curry
def keep(*args, **kwargs):
    """Alias for `filter` with `invert=False`"""
    invert = kwargs.pop("invert", False)
    return filter(*args, invert=invert, **kwargs)


@curry
def discard(*args, **kwargs):
    """Alias for `filter` with `invert=True`"""
    invert = kwargs.pop("invert", True)
    return filter(*args, invert=invert, **kwargs)


def seq(n):
    """Enumerated `list`"""
    return list(range(n))


def equal(*seqs):
    """
    Checks if N args of potentionally different lengths are equal.
    Non-iterable args are directly compared with `==`
    Dataframes and arrays use `.equals()` and `np.allclose()` respectively
    """

    if not isinstance(seqs[0], Iterable):
        return checkall(lambda e: e == seqs[0], seqs)

    if isinstance(seqs[0], pd.DataFrame):
        return checkall(lambda e: e.equals(seqs[0]), seqs)

    if isinstance(seqs[0], np.ndarray):
        return checkall(lambda e: np.allclose(e, seqs[0]), seqs)

    # For other sequence types we can be lazy
    return not any(diff(*seqs, default=object()))


def aslist(e):
    """Idempotently convert something to a list."""
    return [e] if not isinstance(e, list) else e


def asstr(e):
    """Idempotently convert something to a str."""
    return str(e) if not isinstance(e, str) else e


def isempty(iterme: Iterable):
    """Check if iterable is empty"""
    return len(iterme) == 0


@curry
def checkany(func, iterme, transparent=False):
    """
    Check if any elements are `func(elem) == True`

    Args:
        func (callable): function that returns True or False
        iterme (iterable): iterable
        transparent (bool, optional): return iterme instead of result if check passes, useful in pipes; Default False

    Returns:
        bool: True or False
    """

    result = any(map(func, iterme))
    if transparent and result:
        return iterme
    if transparent and not result:
        raise ValueError("Check failed")
    return result


@curry
def checkall(func, iterme, transparent=False):
    """
    Check if all elements are `func(elem) == True`

    Args:
        func (callable): function that returns True or False
        iterme (iterable): iterable
        transparent (bool, optional): return iterme instead of result if check passes, useful in pipes; Default False

    Returns:
        bool: True or False
    """

    result = all(map(func, iterme))
    if transparent and result:
        return iterme
    if transparent and not result:
        raise ValueError("Check failed")
    return result


@curry
def pairs(iterme: Iterable):
    return list(it.combinations(iterme, 2))
