"""
dplyr like *verbs* for working with pandas dataframes. Designed to be piped together using the `pipe` function from the `toolz` package. 

"""

__all__ = [
    "groupby",
    "rows",
    "cols",
    "rename",
    "save",
    "summarize",
    "assign",
    "apply",
    "head",
    "tail",
]

import numpy as np
import pandas as pd
from toolz import curry
from plydata import select, group_by


@curry
def groupby(cols, df, **kwargs):
    """Call a dataframe's `.groupby` method"""
    if kwargs.get("use_ply", False):
        return group_by(df, cols)
    return df.groupby(cols)


@curry
def rows(query, df):
    """Select rows using a `.query` (str), slicerange (start,stop,step), or indices (list)"""
    if isinstance(query, str):
        return df.query(query).reset_index(drop=True)
    elif isinstance(query, (list, np.ndarray)):
        if isinstance(query[0], str):
            return df.loc[query, :]
        return df.iloc[query, :]
    elif isinstance(query, tuple):
        return df.iloc[slice(*query), :]
    elif isinstance(query, int):
        if df.shape[0] - query > 1:
            return df.iloc[query : query + 1, :]
        else:
            return df.iloc[query:, :]


@curry
def cols(query, df, **kwargs):
    """Select columns using a `.query` (str), slicerange (start,stop,step), or indices (list). Uses `plydata.select`"""
    if isinstance(query, str):
        return select(df, query)
    elif isinstance(query, (list, np.ndarray)):
        if isinstance(query[0], str):
            return select(df, *query)
        return df.iloc[:, query]
    elif isinstance(query, tuple):
        return df.iloc[:, slice(*query)]
    elif isinstance(query, int):
        if df.shape[1] - query > 1:
            return df.iloc[:, query : query + 1]
        else:
            return df.iloc[:, query:]


@curry
def rename(cols, df):
    """Call a dataframe's `.rename(columns={})` method"""
    return df.rename(columns=cols)


@curry
def save(path, df):
    """Call a dataframe's `.to_csv(index=False)` method"""
    df.to_csv(f"{path}.csv", index=False)
    return df


@curry
def summarize(df, **stats):
    """Call a dataframe or groupby object's `.agg` method"""
    return df.agg(stats)


@curry
def assign(dfg, *args, **kwargs):
    """
    Creates a new column(s) in a DataFrame based on a function of existing columns in the DataFrame. Uses `plydata.define/mutate` unless the input is a grouped DataFrame
    in which case it falls back to pandas methods because `plydata` can only handle
    grouped inputs resulting from its own (slow) `group_by` function
    """

    if isinstance(dfg, pd.core.groupby.generic.DataFrameGroupBy):
        prev = dfg.filter(lambda _: True).reset_index()
        for _, (k, v) in enumerate(kwargs.items()):
            res = dfg.apply(lambda group: group.eval(v)).reset_index()
            group_col = res.columns[0]
            if isinstance(res, pd.DataFrame) and "level_1" in res.columns:
                res.columns = [res.columns[0], "index"] + [k]
                prev = prev.merge(res.drop(columns=group_col), on="index")
            else:
                res.columns = [res.columns[0]] + [k]
                prev = prev.merge(res, on=group_col)
        prev = prev.sort_values(by="index").drop(columns="index").reset_index(drop=True)
        return prev
    else:
        from plydata import define

        return define(dfg, *args, **kwargs)


@curry
def apply(func, df):
    """Call a dataframe or groupby object's `.apply` method"""
    return df.apply(func)


@curry
def head(df, n=5):
    """Call dataframe's `.head()` method"""
    return df.head(n=n)


@curry
def tail(df, n=5):
    """Call dataframe's `.tail()` method"""
    return df.tail(n=n)
