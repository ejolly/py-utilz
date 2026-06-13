"""Tests for the core pipe API: the `|` operator, `pipeable`, and `Pipe`."""

import polars as pl
import pytest

from utilz import pipe, pipeable, Pipe, append, spread, gather, fork, pop


def test_pipeable_basic():
    """A pipeable wraps a plain function so `data | f` runs `f(data)`."""
    f = pipeable(lambda x: x + 1)
    assert isinstance(f, Pipe)
    assert 5 | f == 6
    # Still directly callable
    assert f(5) == 6


def test_pipeable_decorator():
    """`pipeable` works as a decorator on a one-arg function."""

    @pipeable
    def inc(x):
        return x + 1

    assert 10 | inc == 11
    assert inc.__name__ == "inc"  # update_wrapper preserves metadata


def test_pipe_operator_chain():
    """Multiple steps chain left-to-right with `|`, no pipe() needed."""
    out = [1, 2, 3] | pipeable(lambda xs: [x + 1 for x in xs]) | pipeable(sum)
    assert out == 9


def test_pipe_and_operator_are_equivalent():
    """`data | f1 | f2` matches `pipe(data, f1, f2)`."""
    f1 = pipeable(lambda x: x * 2)
    f2 = pipeable(lambda x: x + 1)
    assert (5 | f1 | f2) == pipe(5, f1, f2) == 11


def test_point_free_composition():
    """Two Pipes compose into a reusable Pipe via `f1 | f2`."""
    combo = pipeable(lambda x: x * 2) | pipeable(lambda x: x + 1)
    assert isinstance(combo, Pipe)
    assert 5 | combo == 11
    assert combo(5) == 11


def test_pipe_requires_callable():
    with pytest.raises(TypeError):
        Pipe(5)


def test_helpers_support_pipe_operator():
    """The built-in helpers return Pipes and work with `|`."""
    # append: (input, result)
    assert (3 | append(lambda x: x * 10)) == (3, 30)
    # fork: duplicate n times
    assert (5 | fork(3)) == (5, 5, 5)
    # spread + gather
    assert ([2] | spread(lambda d: d[0] ** 2, lambda d: d[0] + 1) | gather(lambda a, b: a + b)) == 7
    # pop prunes a tuple element
    assert (3 | append(lambda x: x * 10) | pop(1)) == 3


def test_helpers_still_work_in_pipe():
    """Backwards compatibility: helpers still work inside pipe()."""
    assert pipe(3, append(lambda x: x * 10)) == (3, 30)
    assert pipe(5, fork(3)) == (5, 5, 5)


def test_pipe_operator_with_polars():
    """`|` dispatches correctly for polars DataFrames (NotImplemented fallback)."""
    df = pl.DataFrame({"a": [1, 2, 3]})
    out = df | pipeable(lambda d: d.with_columns((pl.col("a") * 2).alias("b")))
    assert out.columns == ["a", "b"]
    assert out["b"].to_list() == [2, 4, 6]


def test_repr():
    @pipeable
    def clean(x):
        return x

    assert repr(clean) == "Pipe(clean)"
