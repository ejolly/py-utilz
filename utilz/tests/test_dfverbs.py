import pandas as pd
import pytest
from utilz.dfverbs import groupby, apply, head
from utilz import randdf, pipe


def test_groupby():
    df = randdf((15, 3)).assign(Group=["A"] * 5 + ["B"] * 5 + ["C"] * 5)
    out = pipe(df, groupby("Group"), apply(lambda g: g.A1 + g.B1))
    assert out.shape == (15,)
    breakpoint()


def test_head():
    df = randdf()
    out = pipe(df, head)
    assert out.shape == (5, 3)
    out = pipe(df, head(n=2))
    assert out.shape == (2, 3)
    breakpoint()
