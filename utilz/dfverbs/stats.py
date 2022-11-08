"""
dataframe stats methods
"""

__all__ = [
    "mean",
    "median",
    "min",
    "max",
    "mode",
    "var",
    "std",
    "sum",
    "sem",
    "prod",
    "round",
    "abs",
    "sqrt",
    "all",
    "any",
    "corr",
    "cov",
    "count",  # pandas counts na
    "unique",
    "nunique",
    "value_counts",  # dplyr count
    "rank",
    "size",
    "bootci",
]

import numpy as np
import pandas as pd
from toolz import curry
from .verbs import _reset_index_helper, apply, split, summarize, merge, mutate
import seaborn as sns
from ..ops import pipe
from ..maps import filter


@curry
def mean(*args, **kwargs):
    """Call df.mean"""

    def call(df):
        return df.mean(*args, **kwargs)

    return call


@curry
def median(*args, **kwargs):
    """Call df.median"""

    def call(df):
        return df.median(*args, **kwargs)

    return call


@curry
def min(*args, **kwargs):
    """Call df.min"""

    def call(df):
        return df.min(*args, **kwargs)

    return call


@curry
def max(*args, **kwargs):
    """Call df.max"""

    def call(df):
        return df.max(*args, **kwargs)

    return call


@curry
def mode(*args, **kwargs):
    """Call df.mode"""

    def call(df):
        return df.mode(*args, **kwargs)

    return call


@curry
def var(*args, **kwargs):
    """Call df.var"""

    def call(df):
        return df.var(*args, **kwargs)

    return call


@curry
def std(*args, **kwargs):
    """Call df.std"""

    def call(df):
        return df.std(*args, **kwargs)

    return call


@curry
def sum(*args, **kwargs):
    """Call df.sum"""

    def call(df):
        return df.sum(*args, **kwargs)

    return call


@curry
def sem(*args, **kwargs):
    """Call df.sem"""

    def call(df):
        return df.sem(*args, **kwargs)

    return call


@curry
def prod(*args, **kwargs):
    """Call df.prod"""

    def call(df):
        return df.prod(*args, **kwargs)

    return call


@curry
def round(*args, **kwargs):
    """Call df.round"""

    def call(df):
        return df.round(*args, **kwargs)

    return call


@curry
def abs(*args, **kwargs):
    """Call df.abs"""

    def call(df):
        return df.abs(*args, **kwargs)

    return call


@curry
def sqrt(*args, **kwargs):
    """Call df.sqrt"""

    def call(df):
        return df.sqrt(*args, **kwargs)

    return call


@curry
def all(*args, **kwargs):
    """Call df.all"""

    def call(df):
        return df.all(*args, **kwargs)

    return call


@curry
def any(*args, **kwargs):
    """Call df.any"""

    def call(df):
        return df.any(*args, **kwargs)

    return call


@curry
def corr(*args, **kwargs):
    """Call df.corr"""

    def call(df):
        return df.corr(*args, **kwargs)

    return call


@curry
def cov(*args, **kwargs):
    """Call df.cov"""

    def call(df):
        return df.cov(*args, **kwargs)

    return call


@curry
def count(*args, **kwargs):
    """Call df.count"""

    def call(df):
        return df.count(*args, **kwargs)

    return call


@curry
def unique(*args, **kwargs):
    """Call df.unique"""

    def call(df):
        # Unique only exists on series, so squeeze down single col df or loop over cols
        if df.shape[1] == 1:
            out = df.squeeze().unique(*args, **kwargs)
        else:
            out = df.apply(lambda col: col.unique(*args, **kwargs))

        # just one col
        if isinstance(out, np.ndarray):
            out = pd.DataFrame(out, columns=["unique"])
            out["column"] = df.columns[0]
            return out[["column", "unique"]]
        return out.reset_index().rename(columns={"index": "column", 0: "unique"})

    return call


@curry
def nunique(*args, **kwargs):
    """Call df.nunique"""

    def call(df):
        out = df.nunique(*args, **kwargs)
        return out.reset_index().rename(columns={"index": "column", 0: "nunique"})

    return call


@curry
def value_counts(*args, **kwargs):
    """Call df.value_counts"""

    def call(df):
        out = df.value_counts(*args, **kwargs)
        return out.reset_index().rename(columns={"index": "column", 0: "count"})

    return call


@curry
def rank(*args, **kwargs):
    """Call df.rank"""

    def call(df):
        return df.rank(*args, **kwargs)

    return call


@curry
def size(*args, **kwargs):
    """Call df.size"""

    def call(df):
        return df.size(*args, **kwargs)

    return call


@curry
def bootci(col, **kwargs):
    """Calculate 95% bootstrapped confidence intervals on the mean of a column. Unlike
    summarize, bootci expects a string column name and will return a summary frame with
    columns for the mean, 2.5% and 97.% confidence limits. Use `as_devation=True` to
    convert the CIs to deviations from the mean. Accepts all the same args as
    `seaborn.algorithms.bootstrap`, e.g. `units`."""

    deviation = kwargs.pop("as_deviation", False)

    def call(df):
        if isinstance(df, pd.core.groupby.generic.DataFrameGroupBy):
            units = kwargs.pop("units", None)

            cis = pipe(
                df,
                apply(
                    lambda g: sns.utils.ci(
                        sns.algorithms.bootstrap(
                            g[col],
                            units=g[units] if units is not None else None,
                            **kwargs,
                        )
                    ),
                    reset_index="reset",
                ),
                split(0, [f"{col}_ci_l", f"{col}_ci_u"], sep=[]),
            )
            summary = pipe(df, summarize(**{f"{col}_mean": f"{col}.mean()"}))
            matching_cols = filter(summary.columns, cis.columns)
            cis = pipe(cis, merge(summary, on=matching_cols))
            if deviation:
                cis = pipe(
                    cis,
                    mutate(
                        **{
                            f"{col}_ci_l": f"{col}_mean - {col}_ci_l",
                            f"{col}_ci_u": f"{col}_ci_u - {col}_mean",
                        },
                    ),
                )

            return cis
        else:
            raise TypeError(
                "bootci only works on grouped dataframes, trying call _.groupby before"
            )

    return call
