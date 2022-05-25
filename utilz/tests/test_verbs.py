from utilz.verbs import rows, cols, head, tail, rename, assign, summarize, save
from utilz.boilerplate import randdf
from utilz.io import load
from toolz import pipe
import pytest
import numpy as np
from pathlib import Path


def test_rows():

    df = randdf(size=(10, 3))
    df.index = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    df = df.assign(group=["a"] * 5 + ["b"] * 5)

    assert pipe(df, rows(0)).shape == (1, 4)
    assert pipe(df, rows((0, 5))).shape == (5, 4)
    assert pipe(df, rows([0, 5])).shape == (2, 4)
    assert pipe(df, rows(["a", "d", "g"])).shape == (3, 4)
    assert pipe(df, rows("group == 'a'")).shape == (5, 4)


def test_cols():

    df = randdf(size=(10, 10))
    assert pipe(df, cols(0)).shape == (10, 1)
    assert pipe(df, cols((0, 5))).shape == (10, 5)
    assert pipe(df, cols([0, 5])).shape == (10, 2)
    assert pipe(df, cols("A1")).shape == (10, 1)
    assert pipe(df, cols("A1")).shape == (10, 1)
    assert pipe(df, cols("-A1")).shape == (10, 9)
    assert pipe(df, cols(["A1", "C1"])).shape == (10, 2)


def test_head():
    df = randdf()
    assert pipe(df, head).shape == (5, 3)
    assert pipe(df, head()).shape == (5, 3)
    assert pipe(df, head(n=2)).shape == (2, 3)


def test_tail():
    df = randdf()
    assert pipe(df, tail).shape == (5, 3)
    assert pipe(df, tail()).shape == (5, 3)
    assert pipe(df, tail(n=2)).shape == (2, 3)


def test_rename():
    df = randdf()
    assert "new" in pipe(df, rename({"A1": "new"})).columns


@pytest.mark.skip()
def test_apply():
    pass


@pytest.mark.skip()
def test_groupby():
    pass


@pytest.mark.skip()
def test_assign():
    pass


def test_save():
    df = pipe(randdf((20, 3)), save("test"))
    f = Path("test.csv")
    assert f.exists()
    ddf = load("test.csv")
    assert np.allclose(df.to_numpy(), ddf.to_numpy())
    f.unlink()


def test_pipeline():
    # TODO add more steps to pipeline as more tests get made OR add more complicated
    # pipelines
    df = pipe(
        randdf((20, 3)),
        assign(D1=list("abcde") * 4),
        rename({"A1": "rt", "B1": "score", "C1": "speed", "D1": "group"}),
        assign(rt_doubled="rt*2"),
        save("test"),
    )
    f = Path("test.csv")
    assert f.exists()
    assert df.shape == (20, 5)
    assert all(
        map(lambda c: c in df.columns, ["rt", "score", "speed", "group", "rt_doubled"])
    )
    assert np.allclose(df.rt * 2, df.rt_doubled)
    assert df.group.to_list() == list("abcde") * 4
    # Clean up
    f.unlink()
