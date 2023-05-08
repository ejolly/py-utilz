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

    out = pipe(df, _.query(lambda A1: A1 > x))
    assert out.shape[0] < df.shape[0]


def test_mutate():
    df = randdf((20, 3))
    # Assign values directly
    out = pipe(df, _.mutate(group=["A"] * 10 + ["B"] * 10))
    assert "group" in out.columns

    # Or use str eval ops
    out = pipe(out, _.mutate(A1_doubled="A1 * 2"))
    assert all(out.A1 * 2 == out.A1_doubled)

    # Can use functions too, if function takes a single arg "df, g, group" then it's
    # like pandas.assign
    out2 = pipe(out, _.mutate(A1_B1=lambda df: df.A1 + df.B1))
    assert all((out.A1 + out.B1) == out2.A1_B1)

    # Otherwise you can give a lambda with another name and it'll be evaluated against
    # column names, like pandas transform
    out3 = pipe(out, _.mutate(A1_log=lambda A1: np.log(A1)))
    assert all(out.A1.apply(np.log) == out3.A1_log)

    # works with multiple cols too
    out3 = pipe(out, _.mutate(A1_B1=lambda A1, B1: A1 + B1))
    assert all((out.A1 + out.B1) == out3.A1_B1)

    df = pd.read_csv("./utilz/tests/mtcars.csv")

    # broadcast scalar
    out = pipe(df, _.mutate(hp_grp_mean="hp.mean()"))
    assert out.shape[0] == df.shape[0]
    assert out["hp_grp_mean"].nunique() == 1

    # Groupby mutate returns same size as mutate like pandas transform
    out = pipe(df, _.mutate(hp_norm="hp / hp.mean()"))
    out_grouped = pipe(df, _.groupby("cyl"), _.mutate(hp_norm="hp / hp.mean()"))

    assert "hp_norm" in out and "hp_norm" in out_grouped
    assert out.shape == out_grouped.shape
    assert all(out["hp_norm"] == df["hp"] / df["hp"].mean())
    correct = (
        df.groupby("cyl")
        .apply(lambda g: g.hp / g.hp.mean())
        .reset_index()
        .sort_values(by="level_1")["hp"]
        .reset_index(drop=True)
    )
    assert all(out_grouped["hp_norm"] == correct)

    # multiple groups broadcast scalar
    out = pipe(df, _.groupby("cyl", "gear"), _.mutate(hp_grp_mean="hp.mean()"))
    assert out["hp_grp_mean"].nunique() == 8

    # multiple group vector
    out_grouped_two = pipe(
        df, _.groupby("cyl", "gear"), _.mutate(hp_norm="hp / hp.mean()")
    )
    assert out.shape == out_grouped_two.shape
    assert not out_grouped.equals(out_grouped_two)

    # Can also use function shorthand
    out = pipe(
        df, _.groupby("cyl", "gear"), _.mutate(hp_grp_mean=lambda hp: np.mean(hp))
    )
    assert out["hp_grp_mean"].nunique() == 8

    # multiple groups and cols
    out_twogrp = pipe(
        df,
        _.groupby("cyl", "gear"),
        _.mutate(hp_norm_disp_by_cycl_gear=lambda hp, disp: hp / disp),
    )
    correct = (
        df.groupby(["cyl", "gear"])
        .apply(lambda g: g.hp / g.disp)
        .reset_index()
        .sort_values(by="level_2")[0]
        .reset_index(drop=True)
    )
    assert all(out_twogrp["hp_norm_disp_by_cycl_gear"] == correct)

    out_onegrp = pipe(
        df,
        _.groupby("cyl"),
        _.mutate(hp_norm_disp_by_cycl_gear=lambda hp, disp: hp / disp),
    )
    assert all(out_onegrp["hp_norm_disp_by_cycl_gear"] == correct)

    out_nogrp = pipe(
        df,
        _.mutate(hp_norm_disp_by_cycl_gear=lambda hp, disp: hp / disp),
    )
    assert all(out_nogrp["hp_norm_disp_by_cycl_gear"] == correct)
    # Since in this case grouping doesn't affect the function out come we'd expect
    # results to be same as grouping by fewer calls or not using groupby at all
    assert all(
        out_onegrp["hp_norm_disp_by_cycl_gear"]
        == out_twogrp["hp_norm_disp_by_cycl_gear"]
    ) and all(
        out_twogrp["hp_norm_disp_by_cycl_gear"]
        == out_nogrp["hp_norm_disp_by_cycl_gear"]
    )

    out_asstr = pipe(
        df,
        _.groupby("cyl"),
        _.mutate(hp_norm="hp - hp.mean()", disp_norm="disp-disp.mean()"),
    )
    assert "hp_norm" in out_asstr and "disp_norm" in out_asstr

    out_asfunc = pipe(
        df,
        _.groupby("cyl"),
        _.mutate(
            hp_norm=lambda hp: hp - hp.mean(), disp_norm=lambda disp: disp - disp.mean()
        ),
    )
    assert "hp_norm" in out_asfunc and "disp_norm" in out_asfunc

    assert out_asstr.equals(out_asfunc)


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

    # str
    out = pipe(df, _.summarize(avg="mpg.mean()"))
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (1, 1)

    # funcs, where if lambda takes 'df, g, group' will give lamba entire frame
    out = pipe(df, _.summarize(avg="mpg.mean()", n=lambda df: df.shape[0]))
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (1, 2)
    assert np.allclose(out.values, np.array([[20.090625, 32.0]]))

    # Can also use function shorthand
    out = pipe(df, _.summarize(avg=lambda mpg: mpg.mean(), n=lambda df: df.shape[0]))
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

    out2 = pipe(
        df,
        _.groupby("cyl", "gear"),
        _.summarize(mean=lambda disp: disp.mean(), n=lambda g: g.shape[0]),
    )
    assert out2.equals(out)

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
        _.mutate(
            dimorphism=lambda male, female: np.round(male / female, 2)
        ),  # instead use a func
    )
    assert out.shape == (16, 6)
    assert all(out.columns == ["bear", "genus", "stat", "female", "male", "dimorphism"])

    # Test bootstrap ci
    boots = pipe(
        out,
        _.drop("stat"),
        _.pivot_longer(["male", "female"], into=("sex", "weight")),
        _.call("drop_duplicates"),
        _.groupby("bear"),
        _.bootci("weight", n_boot=100, seed=0),
    )
    assert equal(boots.columns, ["bear", "weight_mean", "weight_ci_l", "weight_ci_u"])
    assert all(boots["weight_ci_l"] < boots["weight_mean"])
    assert all(boots["weight_ci_u"] > boots["weight_mean"])

    # As deviations for plotting libs, e.g. matplotlib or plotly
    boots2 = pipe(
        out,
        _.drop("stat"),
        _.pivot_longer(["male", "female"], into=("sex", "weight")),
        _.call("drop_duplicates"),
        _.groupby("bear"),
        _.bootci("weight", as_deviation=True, n_boot=100, seed=0),
    )
    assert all(boots2["weight_ci_l"] == boots["weight_mean"] - boots["weight_ci_l"])
    assert all(boots2["weight_ci_u"] == boots["weight_ci_u"] - boots["weight_mean"])
