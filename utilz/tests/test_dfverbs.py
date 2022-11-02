import utilz.dfverbs as _
from utilz import randdf, pipe, equal
import numpy as np
import pandas as pd
import pytest


def test_head():
    df = randdf()
    assert pipe(df, _.head()).equals(df.head())
    assert pipe(df, _.head(3)).equals(df.head(3))
    assert pipe(df, _.head(n=10)).equals(df.head(10))


def test_tail():
    df = randdf()
    assert pipe(df, _.tail()).equals(df.tail())
    assert pipe(df, _.tail(3)).equals(df.tail(3))
    assert pipe(df, _.tail(n=10)).equals(df.tail(10))


def test_apply():
    df = randdf()
    assert pipe(df, _.apply(np.sqrt)).equals(df.apply(np.sqrt))
    df = randdf((20, 3), groups={"group": 4})
    out = pipe(
        df,
        _.groupby("group"),
        _.apply(lambda g: g.A1 - g.A1.mean(), reset_index="none"),
    )
    assert out.equals(df.groupby("group").apply(lambda g: g.A1 - g.A1.mean()))


def test_query():
    df = randdf()
    out = pipe(df, _.query("A1 > 0.5"))
    assert out.shape[0] < df.shape[0]

    x = 0.5
    out = pipe(df, _.query(lambda df: df.A1 > x))
    assert out.shape[0] < df.shape[0]


def test_mutate():

    df = pd.read_csv("./utilz/tests/mtcars.csv")

    out = pipe(df, _.mutate(hp_norm="hp / hp.mean()"))

    out2 = pipe(df, _.groupby("cyl"), _.mutate(hp_norm="hp / hp.mean()"))

    assert out.shape == out2.shape
    assert "hp_norm" in out and "hp_norm" in out2
    assert not out.equals(out2)

    # multiple groups
    out3 = pipe(df, _.groupby("cyl", "gear"), _.mutate(hp_norm="hp / hp.mean()"))
    assert out.shape == out3.shape
    assert not out2.equals(out3)

    # broadcast scalar
    out = pipe(df, _.groupby("cyl", "gear"), _.mutate(hp_grp_mean="hp.mean()"))
    assert out["hp_grp_mean"].nunique() == 8

    df = randdf((20, 3))

    # Assign values
    out = pipe(df, _.mutate(group=["A"] * 10 + ["B"] * 10))
    assert "group" in out.columns

    # Or using functions
    out = pipe(out, _.mutate(A1_doubled=lambda df: df.A1 * 2))
    assert all(out.A1 * 2 == out.A1_doubled)

    # Or using strs (also tests drop)
    out = pipe(out, _.drop("A1_doubled"), _.mutate(A1_doubled="A1 * 2"))
    assert all(out.A1 * 2 == out.A1_doubled)


def test_select():
    df = randdf()

    # Always returns a dataframe
    out = pipe(df, _.select("A1"))
    assert out.equals(pd.DataFrame(df["A1"]))

    out = pipe(df, _.select("-B1"))
    assert equal(out.columns, ["A1", "C1"])
    out2 = pipe(df, _.select("A1", "C1"))
    assert equal(out, out2)

    df = randdf((20, 3), groups={"group": 4})
    out = pipe(df, _.groupby("group"), _.select("A1"), _.call("mean"))
    assert out.equals(df.groupby("group")["A1"].mean())


def test_summarize():

    df = pd.read_csv("./utilz/tests/mtcars.csv")

    out = pipe(df, _.summarize(avg="mpg.mean()"))
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (1, 1)

    out = pipe(df, _.summarize(avg="mpg.mean()", n=lambda df: df.shape[0]))
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (1, 2)
    assert np.allclose(out.values, np.array([[20.090625, 32.0]]))

    # With groupby
    out = pipe(
        df, _.groupby("cyl"), _.summarize(mean="disp.mean()", n=lambda g: g.shape[0])
    )
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (3, 3)

    out = pipe(
        df,
        _.groupby("cyl", "gear"),
        _.summarize(mean="disp.mean()", n=lambda g: g.shape[0]),
    )
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (8, 4)

    # Expressions and opertions to summarize should return smaller df
    with pytest.raises(ValueError):
        out = pipe(
            df,
            _.groupby("cyl", "gear"),
            _.summarize(mean="disp - disp.mean()"),
        )

    # Summarize doesn't work after select
    with pytest.raises(TypeError):
        out = pipe(
            df,
            _.groupby("cyl", "gear"),
            _.select("disp"),
            _.summarize(mean="disp.mean"),
        )


# Also tests advanced mutate
def test_groupby():
    df = randdf((20, 3))
    out = pipe(
        df,
        _.mutate(
            group=["A"] * 10 + ["B"] * 10,
            school=["a"] * 5 + ["b"] * 5 + ["c"] * 5 + ["d"] * 5,
        ),
    )

    # Single group
    groups = pipe(out, _.groupby("group"))
    assert isinstance(groups, pd.core.groupby.generic.DataFrameGroupBy)

    # Grouped create 1 new col
    groups = pipe(
        out, _.groupby("group"), _.mutate(A1_demean_by_group="A1 - A1.mean()")
    )
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
        _.groupby("group"),
        _.mutate(A1_group_mean="A1.mean()", A1_demean_by_group="A1 - A1.mean()"),
    )
    assert groups.shape[0] == groups.shape[0]
    assert groups["A1_group_mean"].nunique() == 2
    assert np.allclose(
        groups["A1_demean_by_group"].to_numpy(),
        out.groupby("group").A1.transform(lambda c: c - c.mean()),
    )

    # Nested groups
    schools = pipe(out, _.groupby("group", "school"))
    assert isinstance(schools, pd.core.groupby.generic.DataFrameGroupBy)

    # Nested assign
    schools = pipe(
        out,
        _.groupby("group", "school"),
        _.mutate(
            B1_mean_by_group_and_school="B1.mean()",
        ),
    )
    assert schools.shape[0] == out.shape[0]
    assert "B1_mean_by_group_and_school" in schools.columns
    assert schools["B1_mean_by_group_and_school"].nunique() == 4

    # Nested multi-assign
    schools = pipe(
        out,
        _.groupby("group", "school"),
        _.mutate(
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
        _.groupby("group", "school"),
        _.mutate(
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


def test_sort():

    data = randdf((20, 3), groups={"condition": 2, "group": 4})

    out = pipe(data, _.sort("group", "condition", ascending=False))
    assert out.equals(
        data.sort_values(by=["group", "condition"], ascending=False, ignore_index=True)
    )

    out = pipe(
        data,
        _.groupby("group"),
        _.mutate(A1_sorted_by_group="A1.sort_values()"),
    )
    assert "A1_sorted_by_group" in out


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
        _.rename({"weight (male, lbs)": "male", "weight (female, lbs)": "female"}),
        _.pivot_longer(columns=["male", "female"], into=("sex", "weight")),
        _.split("weight", ("min", "max"), sep="-"),
        _.pivot_longer(columns=["min", "max"], into=("stat", "weight")),
        _.astype({"weight": float}),
        _.groupby("genus", "sex"),
        _.mutate(weight="weight.mean()"),
        _.pivot_wider(column="sex", using="weight"),
        _.mutate(dimorphism="male / female"),  # no rounding possible
        _.mutate(dimorphism=lambda df: np.round(df.male / df.female, 2)),
    )
    assert out.shape == (16, 6)
    assert all(out.columns == ["bear", "genus", "stat", "female", "male", "dimorphism"])
