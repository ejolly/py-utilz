"""Shorthands for common operations or ops with specific params set"""

__all__ = ["keep", "discard", "seq", "equal"]

from .maps import filter
from .ops import curry
from toolz import diff


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
    """Lazily checks if two sequences of potentionally different lengths are equal"""
    return not any(diff(*seqs, default=object()))


def aslist(e):
    """Idempotently convert something to a list."""
    return [e] if not isinstance(e, list) else e


def asstr(e):
    """Idempotently convert something to a str."""
    return str(e) if not isinstance(e, str) else e
