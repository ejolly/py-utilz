from utilz.shorts import checkany, checkall, seq, equal, isempty, equal
from utilz.pipes import pipe
import pandas as pd
import numpy as np
from utilz.boilerplate import randdf
import pytest


def test_checks():
    # Basic boolean check
    l = seq(10)
    assert checkany(lambda x: x < 5, l)
    assert not checkall(lambda x: x < 5, l)
    assert checkall(lambda x: x < 10, l)

    # Currying
    assert pipe(l, checkany(lambda x: x < 5))
    assert not pipe(l, checkall(lambda x: x < 5))
    assert pipe(l, checkany(lambda x: x < 10))

    # Transparency in pips returns the result if check passes
    assert equal(l, pipe(l, checkany(lambda x: x < 5, transparent=True)))
    assert equal(l, pipe(l, checkany(lambda x: x < 10, transparent=True)))

    # Otherwise raises value error
    with pytest.raises(ValueError):
        pipe(l, checkall(lambda x: x < 5, transparent=True))


def test_isempty():
    assert isempty(pd.DataFrame())
    assert not isempty(randdf())

    assert isempty(np.array([]))
    assert not isempty(np.arange(10))

    assert isempty(dict())
    assert not isempty({"name": "name"})


def test_equal():
    # Compare scalars
    x, y = 2, 2
    assert equal(x, y)

    # And strings
    x, y = "hi", "bye"
    assert not equal(x, y)

    # Compare dataframes
    assert not equal(randdf(), randdf())

    df = randdf()
    assert equal(df, df.copy())

    # Compare numpy arrays
    data = [np.arange(10).astype(float) for i in range(4)]
    # We can compare as many items as we want, equivalent to:
    # equal(data[0], data[1], data[2])
    assert equal(*data)

    # numpy comparisons are made using np.allclose
    data[0] += 0.00001
    assert not equal(*data)
