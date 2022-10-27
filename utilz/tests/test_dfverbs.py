from utilz.dfverbs import (
    head,
    tail,
    apply,
    query,
    assign,
    groupby,
    drop,
    select,
    summarize,
)
from utilz import randdf, pipe, equal
import numpy as np
import pandas as pd


def test_head():
    df = randdf()
    assert pipe(df, head).equals(df.head())
    assert pipe(df, head(n=10)).equals(df.head(10))


def test_tail():
    df = randdf()
    assert pipe(df, tail).equals(df.tail())
    assert pipe(df, tail(n=10)).equals(df.tail(10))


def test_apply():
    df = randdf()
    assert pipe(df, apply(np.sqrt)).equals(df.apply(np.sqrt))


def test_query():
    df = randdf()
    out = pipe(df, query("A1 > 0.5"))
    assert out.shape[0] < df.shape[0]


def test_assign():

    df = randdf((20, 3))

    # Assign values
    out = pipe(df, assign(group=["A"] * 10 + ["B"] * 10))
    assert "group" in out.columns

    # Or using functions
    out = pipe(out, assign(A1_doubled=lambda df: df.A1 * 2))
    assert all(out.A1 * 2 == out.A1_doubled)

    # Or using strs (also tests drop)
    out = pipe(out, drop("A1_doubled"), assign(A1_doubled="A1 * 2"))
    assert all(out.A1 * 2 == out.A1_doubled)


def test_select():
    df = randdf()

    out = pipe(df, select("A1"))
    assert out.equals(pd.DataFrame(df["A1"]))

    out = pipe(df, select("-B1"))
    assert equal(out.columns, ["A1", "C1"])
    out2 = pipe(df, select("A1", "C1"))
    assert equal(out, out2)


def test_groupby():
    df = randdf((20, 3))
    out = pipe(
        df,
        assign(
            group=["A"] * 10 + ["B"] * 10,
            school=["a"] * 5 + ["b"] * 5 + ["c"] * 5 + ["d"] * 5,
        ),
    )

    # Single group
    groups = pipe(out, groupby("group"))
    assert isinstance(groups, pd.core.groupby.generic.DataFrameGroupBy)
    # Grouped assign
    groups = pipe(groups, assign(A1_mean_by_group="A1.mean()"))
    assert "A1_mean_by_group" in groups.columns
    assert groups["A1_mean_by_group"].nunique() == 2

    # Nested groups
    schools = pipe(out, groupby("group", "school"))
    assert isinstance(schools, pd.core.groupby.generic.DataFrameGroupBy)
    # Nested assign
    schools = pipe(
        schools,
        assign(
            A1_mean_by_group_and_school="A1.mean()",
            B1_mean_by_group_and_school="B1.mean()",
        ),
    )
    assert "A1_mean_by_group_and_school" in schools.columns
    assert "B1_mean_by_group_and_school" in schools.columns
    assert schools["B1_mean_by_group_and_school"].nunique() == 4

    # Standard aggregate
    summ = pipe(
        out, groupby("group"), select("-B1"), summarize("mean", "std", tidy=False)
    )
    assert summ.shape == (2, 4)
    assert equal(summ.index, ["A", "B"])

    # More useful to auto-tidy
    summ = pipe(out, groupby("group"), select("-B1"), summarize("mean", "std"))
    assert equal(summ.columns, ["group", "column", "stat", "value"])
    assert summ.shape == (8, 4)

    # WIP for multiple groups

    # summ = pipe(
    #     out, groupby("group", "school"), select("-B1"), summarize("mean", "std")
    # )
    # assert equal(summ.columns, ["group", "school", "column", "stat", "value"])
    # assert summ.shape == (32, 5)
    # # Works with dicts too
    # summ = pipe(
    #     out,
    #     groupby("group"),
    #     summarize(
    #         {"A1": ["mean", "std"], "B1": ["size", "std"], "C1": ["mean", "var"]},
    #     ),
    # )
    # breakpoint()

    # Doesn't quite work yet because first grouping factor is in wide format
    # summ = pipe(
    #     out,
    #     groupby("group", "school"),
    #     summarize(
    #         {"A1": ["mean", "std"], "B1": ["size", "std"], "C1": ["mean", "var"]},
    #     ),
    # )
    # assert summ.shape[1] == 5
