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
    rename,
    to_long,
    to_wide,
    split,
    astype,
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

    x = 0.5
    out = pipe(df, query(lambda df: df.A1 > x))
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


def test_summarize():
    df = randdf()

    # All columns
    out = pipe(df, summarize("mean", "std"))
    assert all(out.columns == ["column", "stat", "value"])
    assert out.shape[0] < df.shape[0]

    # Some columns with select
    out = pipe(
        df,
        select("-B1"),
        summarize("mean", "std"),
    )
    assert out.shape[0] == 4
    assert out["stat"].nunique() == 2
    assert out["column"].nunique() == 2

    # Columns with summarize directly
    out = pipe(
        df,
        summarize(A1=["mean", "std"], B1=["size", "std"], C1=["mean", "var"]),
    )
    assert out.shape[0] == 6
    assert out["stat"].nunique() == 4
    assert out["column"].nunique() == 3


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

    # Grouped create 1 new col
    groups = pipe(out, groupby("group"), assign(A1_demean_by_group="A1 - A1.mean()"))
    assert groups.shape[0] == out.shape[0]
    assert "A1_demean_by_group" in groups.columns
    correct = (
        out.groupby("group", group_keys=False)
        .select("A1")
        .apply(lambda c: c - c.mean())
        .to_numpy(),
    )
    assert np.allclose(groups["A1_demean_by_group"].to_numpy(), correct)

    # Multiple new cols
    groups = pipe(
        out,
        groupby("group"),
        assign(A1_group_mean="A1.mean()", A1_demean_by_group="A1 - A1.mean()"),
    )
    assert groups.shape[0] == groups.shape[0]
    assert groups["A1_group_mean"].nunique() == 2
    assert np.allclose(
        groups["A1_demean_by_group"].to_numpy(),
        out.groupby("group").A1.transform(lambda c: c - c.mean()),
    )

    # Nested groups
    schools = pipe(out, groupby("group", "school"))
    assert isinstance(schools, pd.core.groupby.generic.DataFrameGroupBy)

    # Nested assign
    schools = pipe(
        out,
        groupby("group", "school"),
        assign(
            B1_mean_by_group_and_school="B1.mean()",
        ),
    )
    assert schools.shape[0] == out.shape[0]
    assert "B1_mean_by_group_and_school" in schools.columns
    assert schools["B1_mean_by_group_and_school"].nunique() == 4

    # Nested multi-assign
    schools = pipe(
        out,
        groupby("group", "school"),
        assign(
            A1_mean_by_group_and_school="A1.mean()",
            B1_mean_by_group_and_school="B1.mean()",
        ),
    )
    assert schools.shape[0] == out.shape[0]
    assert "A1_mean_by_group_and_school" in schools.columns
    assert "B1_mean_by_group_and_school" in schools.columns
    assert schools["B1_mean_by_group_and_school"].nunique() == 4

    # Nested multi-assign with mixed aggregations
    schools = pipe(
        out,
        groupby("group", "school"),
        assign(
            A1_mean_by_group_and_school="A1.mean()",
            B1_demeaned_by_group_and_school="B1 - B1.mean()",
        ),
    )
    assert schools.shape[0] == out.shape[0]
    assert "A1_mean_by_group_and_school" in schools.columns
    assert "B1_demeaned_by_group_and_school" in schools.columns
    assert np.allclose(
        schools["B1_demeaned_by_group_and_school"].to_numpy(),
        out.groupby(["group", "school"])["B1"]
        .transform(lambda x: x - x.mean())
        .to_numpy(),
    )

    # Groupby with summarize
    summ = pipe(
        out, groupby("group"), select("-B1"), summarize("mean", "std", tidy=False)
    )
    assert summ.shape == (2, 4)
    assert equal(summ.index, ["A", "B"])

    # More useful to auto-tidy
    summ = pipe(out, groupby("group"), select("-B1"), summarize("mean", "std"))
    assert equal(summ.columns, ["group", "column", "stat", "value"])
    assert summ.shape == (8, 4)

    # Works with dicts too
    summ = pipe(
        out,
        groupby("group"),
        summarize(
            {"A1": ["mean", "std"], "B1": ["size", "std"], "C1": ["mean", "var"]},
        ),
    )
    assert summ.shape[0] == 3 * 2 * 2
    assert summ["stat"].nunique() == 4
    assert summ["column"].nunique() == 3
    assert summ["group"].nunique() == 2

    # Nested groups
    summ = pipe(
        out, groupby("group", "school"), select("-B1"), summarize("mean", "std")
    )
    assert equal(summ.columns, ["group", "school", "column", "stat", "value"])
    assert summ.shape == (16, 5)
    assert summ["group"].nunique() == 2
    assert summ["school"].nunique() == 4
    assert summ["column"].nunique() == 2

    # Nested + dict
    summ = pipe(
        out,
        groupby("group", "school"),
        summarize(
            {"A1": ["mean", "std"], "B1": ["size", "std"], "C1": ["mean", "var"]},
        ),
    )

    assert equal(summ.columns, ["group", "school", "column", "stat", "value"])
    assert summ["stat"].nunique() == 4
    assert summ["column"].nunique() == 3
    assert summ["group"].nunique() == 2
    assert summ["school"].nunique() == 4


def test_rf_pipeline():
    """Test data pipeline demo'd on https://github.com/maxhumber/redframes"""


df = pd.DataFrame(
    {
        "bear": [
            "Brown bear",
            "Polar bear",
            "Asian black bear",
            "American black bear",
            "Sun bear",
            "Sloth bear",
            "Spectacled bear",
            "Giant panda",
        ],
        "genus": [
            "Ursus",
            "Ursus",
            "Ursus",
            "Ursus",
            "Helarctos",
            "Melursus",
            "Tremarctos",
            "Ailuropoda",
        ],
        "weight (male, lbs)": [
            "300-860",
            "880-1320",
            "220-440",
            "125-500",
            "60-150",
            "175-310",
            "220-340",
            "190-275",
        ],
        "weight (female, lbs)": [
            "205-455",
            "330-550",
            "110-275",
            "90-300",
            "45-90",
            "120-210",
            "140-180",
            "155-220",
        ],
    }
)

out = pipe(
    df,
    rename({"weight (male, lbs)": "male", "weight (female, lbs)": "female"}),
    to_long(columns=["male", "female"], into=("sex", "weight")),
    split("weight", ("min", "max"), sep="-"),
    to_long(columns=["min", "max"], into=("stat", "weight")),
    astype({"weight": float}),
    groupby("genus", "sex"),
    assign(weight="weight.mean()"),
    to_wide(column="sex", by="weight"),
    assign(dimorphism="male / female"),  # no rounding possible
    assign(dimorphism=lambda df: np.round(df.male / df.female, 2)),
)
assert out.shape == (16, 6)
assert all(out.columns == ["bear", "genus", "stat", "female", "male", "dimorphism"])
