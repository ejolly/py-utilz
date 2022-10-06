# We just import to ensure custom registered methods are in the namespace; not necessary
# in usage
import utilz.dftools
import pandas as pd
import pytest


def test_norm_by_group(df):
    out = df.norm_by_group("species", "sepal_length")
    assert isinstance(out, pd.DataFrame)
    # Make sure column was added
    assert out.shape[1] > df.shape[1]
    assert "sepal_length_normed_by_species" in out.columns

    # Check multiple columns
    out = df.norm_by_group("species", ["sepal_length", "petal_width"])
    assert isinstance(out, pd.DataFrame)
    # Make sure column was added
    assert out.shape[1] > df.shape[1]
    assert "sepal_length_normed_by_species" in out.columns
    assert "petal_width_normed_by_species" in out.columns

    out = df.norm_by_group("species", "sepal_length", scale=False)
    assert "sepal_length_centered_by_species" in out.columns

    out = df.norm_by_group("species", "sepal_length", center=False)
    assert "sepal_length_scaled_by_species" in out.columns

    out = df.norm_by_group("species", "sepal_length", addcol=False)
    assert isinstance(out, pd.Series)
    assert len(out) == df.shape[0]


def test_assert_balanced_groups(df):

    assert df.assert_balanced_groups("species")

    assert df.assert_balanced_groups("species", 50)

    with pytest.raises(AssertionError):
        df.iloc[:-1, :].assert_balanced_groups("species")


def test_assert_same_nunique(df):

    with pytest.raises(AssertionError):
        assert df.assert_same_nunique("species", "sepal_length")

    df["val"] = list(range(10)) * 15
    assert df.assert_same_nunique("species", "val")


def test_select(df):
    num_cols = df.shape[1]
    out = df.select("species")
    assert out.shape == (df.shape[0], 1)

    out = df.select("sepal_width", "petal_width")
    assert out.shape == (df.shape[0], 2)

    out = df.select("-sepal_width")
    assert out.shape[1] == num_cols - 1

    out = df.select(sepal_width="sepal", petal_width="petal")
    assert out.shape == (df.shape[0], 2)
    assert list(out.columns) == ["sepal", "petal"]

    out = (
        df.groupby("species").select("sepal_width", "petal_width").agg(("mean", "std"))
    )
    aggd = df.groupby("species").agg(
        {"sepal_width": ["mean", "std"], "petal_width": ["mean", "std"]}
    )
    assert out.equals(aggd)

    # Can't mix and match args and kwargs
    with pytest.raises(ValueError):
        df.select("-species", sepal_width="width")

    # Ensure that single selects with groupby objects return series and not dataframe
    # groupby objects just like normal [] indexing
    dfg = df.groupby("species").select("sepal_width")
    assert isinstance(dfg, pd.core.groupby.generic.SeriesGroupBy)

    # This gives us dataframe groupby
    dfg = df.groupby("species").select("sepal_width", "petal_width")
    assert isinstance(dfg, pd.core.groupby.generic.DataFrameGroupBy)
    # Dataframe has no unique method; only Series do
    with pytest.raises(AttributeError):
        dfg.unique()

    # Support negative col naming with groupby objects
    aggd = df.groupby("species").select("-sepal_width").mean()
    assert aggd.shape == (3, 3)
    assert all(aggd.index.tolist() == df.species.unique())

    dfg = df.groupby("species").select("-sepal_length", "-sepal_width", "-petal_length")
    assert isinstance(dfg, pd.core.groupby.generic.SeriesGroupBy)
    dfg.unique()

    dfg = df.groupby("species").select("-sepal_length", "-sepal_width")
    assert isinstance(dfg, pd.core.groupby.generic.DataFrameGroupBy)

    # Dataframe has no unique method; only Series do
    with pytest.raises(AttributeError):
        dfg.unique()


def test_reshape(df):

    # We need a unique identifier
    df = df.reset_index().rename(columns={"index": "pid"})

    # Wide -> Long
    # just uses pd.melt
    long = df.to_long(["pid", "species"], value_name="length")
    assert long.shape == (600, 4)
    assert "length" in long.columns

    # Long -> Wide
    # wraps and slightly modifies pd.pivot
    wide = long.to_wide(index=["pid", "species"], columns="variable").reset_index()
    # Currently not working cause returns columns that are multi-indexed
    # Can't just pass single val for index cause complains about duplicates
    # Some attempts at flattening cols:
    # wide.columns = wide.columns.get_level_values(0)
    # wide.columns = wide.columns.get_level_values(1)
    # wide.columns = wide.columns.get_level_values(1) + wide.columns.get_level_values(0)