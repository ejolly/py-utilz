"""
dplyr like *verbs* for working with pandas dataframes.

"""

__all__ = [
    "groupby",
    "rename",
    "to_csv",
    "summarize",
    "assign",
    "query",
    "apply",
    "head",
    "tail",
    "drop",
    "select",
    "to_long",
    "to_wide",
    "split",
    "astype",
]

import numpy as np
import pandas as pd
from toolz import curry
from .ops import do, filtercat, mapcat


def _reset_index_helper(out, reset_index):
    if reset_index == "drop":
        return out.reset_index(drop=True)
    if reset_index == "reset":
        return out.reset_index()
    return out


@curry
def groupby(*args):
    """Call a dataframe's `.groupby` method"""

    def call(df):
        return do("groupby", df, [*args])

    return call


@curry
def rename(cols, df):
    """Call a dataframe's `.rename(columns={})` method"""
    if isinstance(cols, tuple):
        cols = {cols[0]: cols[1]}
    return df.rename(columns=cols)


@curry
def to_csv(path, df):
    """Call a dataframe's `.to_csv(index=False)` method"""
    df.to_csv(f"{path}.csv", index=False)
    return df


@curry
def summarize(*args, **kwargs):
    """Call a dataframe or groupby object's `.agg` method"""

    tidy = kwargs.pop("tidy", True)

    def call(df):
        if len(args) == 1 and isinstance(args[0], dict):
            out = df.agg(args[0])
        elif len(args) == 0:
            out = df.agg(kwargs)
        else:
            out = df.agg((args))
        if tidy:
            if isinstance(df, pd.core.groupby.generic.DataFrameGroupBy):
                num_grps = len(list(df.groups.keys())[0])
                unstacker = list(range(num_grps))
                out = out.unstack(unstacker)
                out = out.reset_index().rename(
                    columns={"level_0": "column", "level_1": "stat", 0: "value"}
                )
                if len(args) and isinstance(args[0], str):
                    out = out.assign(
                        stat=np.repeat(args, int(out.shape[0] / len(args)))
                    )

                # Rearrange cols
                cols = list(out.columns)
                created, groups = filtercat(
                    ["column", "stat", "value"], cols, invert="split"
                )
                new_order = groups + created
                out = (
                    out[new_order]
                    .dropna()
                    .sort_values(by=groups)
                    .reset_index(drop=True)
                )
            else:
                out = (
                    out.reset_index()
                    .rename(columns={"index": "stat"})
                    .melt(id_vars="stat", var_name="column")
                )
                out = out[["column", "stat", "value"]].dropna().reset_index(drop=True)

            # Drop the column column if it only has one value, e.g. if we did a
            # summarize on a long-form and provided kwarg to summarize
            if "column" in out and out["column"].nunique() == 1:
                out = out.drop(columns="column")
            return out

        else:
            return out

    return call


@curry
def assign(dfg, *args, **kwargs):
    """
    Creates a new column(s) in a DataFrame based on a function of existing columns in the DataFrame. Uses `plydata.define/mutate` unless the input is a grouped DataFrame
    in which case it falls back to pandas methods because `plydata` can only handle
    grouped inputs resulting from its own (slow) `group_by` function
    """

    eval = kwargs.pop("eval", True)
    if isinstance(dfg, pd.core.groupby.generic.DataFrameGroupBy):
        prev = dfg.obj.copy()
        for _, (k, v) in enumerate(kwargs.items()):
            if isinstance(v, str):
                res = dfg.apply(lambda group: group.eval(v)).reset_index()
            elif callable(v):
                res = dfg.apply(v).reset_index()
            # Calling an operation that returns df the same size as the original df,
            # like transform, e.g. 'A1 - A1.mean()'
            if res.shape[0] == prev.shape[0]:
                keep_cols = [col for col in res.columns if not col.startswith("level_")]
                res = res[keep_cols]
                res = res.rename(columns={res.columns[-1]: k})

                # Join on index cause same shape
                # Allow column overwriting
                if k in prev:
                    prev = prev.drop(columns=k).join(res[k])
                else:
                    prev = prev.join(res[k])
            else:
                # otherwise operation returns smaller
                # so we need to join on the grouping col which is the name of the first
                # col in the output
                res = res.rename(columns={res.columns[-1]: k})
                # Allow column overwriting
                if k in prev:
                    prev = prev.drop(columns=k).merge(
                        res, on=res.columns[:-1].to_list()
                    )
                else:
                    prev = prev.merge(res, on=res.columns[:-1].to_list())
        return prev
    else:

        if eval and any(map(lambda e: isinstance(e, str), kwargs.values())):
            out = dfg.copy()
            for (
                k,
                v,
            ) in kwargs.items():
                out = out.assign(**{k: dfg.eval(v)})
            return out
        else:
            return do("assign", dfg, *args, **kwargs)


@curry
def query(*queries, **kwargs):
    """
    Call a dataframe or groupby object's `.query` method. Resets and drops index by
    default. Change this with reset_index='drop'|'reset'|'none'
    """
    reset_index = kwargs.pop("reset_index", "drop")

    def call(df):
        for q in queries:
            if isinstance(q, str):
                df = df.query(q, **kwargs)
            elif callable(query):
                df = df.loc[q]

        return _reset_index_helper(df, reset_index)

    return call


@curry
def apply(func, df, **kwargs):
    """Call a dataframe or groupby object's `.apply` method"""
    return do("apply", df, func, **kwargs)


@curry
def head(df, **kwargs):
    """Call dataframe's `.head()` method"""
    return do("head", df, **kwargs)


@curry
def tail(df, **kwargs):
    """Call dataframe's `.tail()` method"""
    return do("tail", df, **kwargs)


@curry
def drop(*args):
    """Call a dataframe's `.drop(axis=1)` method"""

    def call(df):
        return do("drop", df, [*args], axis=1)

    return call


@curry
def select(*args):
    """
    Select one ore more columns by name. Drop one or more columns by prepending '-' to
    the name. Does not support renaming"""

    def call(df):
        return do("select", df, *args)
        # if isinstance(df, pd.core.groupby.generic.DataFrameGroupBy):
        #     do('select', df, *args)
        # # Get col via name or exclude -name
        # col_list = [*args]
        # # Split columns to keep and drop based on '-' prefix
        # drop, keep = filtercat("-", col_list, invert="split", assert_notempty=False)
        # # Remove the prefix
        # if len(drop):
        #     drop = mapcat(lambda col: col[1:], drop)
        # if len(keep):
        #     return df.drop(columns=drop).filter(items=keep, axis="columns")
        # return df.drop(columns=drop)

    return call


@curry
def to_wide(df, column=None, by=None, drop_index=True):
    """
    Select one ore more columns by name. Drop one or more columns by prepending '-' to
    the name. Does not support renaming"""

    return df.to_wide(column=column, by=by, drop_index=drop_index)


@curry
def to_long(df, columns=None, into=None, drop_index=True):
    """
    Select one ore more columns by name. Drop one or more columns by prepending '-' to
    the name. Does not support renaming"""

    return df.to_long(columns=columns, into=into, drop_index=drop_index)


@curry
def split(col, into, df, sep=" "):
    """Split values in single df column into multiple columns by separator, e.g.
    First-Last -> [First], [Last]. To split list elements use [] as the sep, e.g.
    [1,2,3] -> [1], [2], [3]"""

    if isinstance(sep, str):
        out = df[col].str.split(sep, expand=True)
    elif isinstance(sep, list):
        out = pd.DataFrame(df[col].to_list())
    if len(into) != out.shape[1]:
        raise ValueError(
            f"into has {len(into)} elements, but splitting creates a dataframe with {out.shape[1]} columns"
        )
    else:
        out.columns = list(into)

    return pd.concat([df.drop(columns=col), out], axis=1)


@curry
def astype(coldict, df):
    return df.astype(coldict)
