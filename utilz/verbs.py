"""
dplyr like *verbs* for working with pandas dataframes. Designed to be piped together using the `pipe` function from the `toolz` package. While not being as feature-complete, `utilz.verbse` provides limited alternative to other libraries like `plydata` because it just wraps mostly native pandas methods under-the-hood. Note: the current version unfortunately does borrow the `select` and `define` functions from plydata

"""

__all__ = ["groupby", "rows", "cols", "rename", "save", "summarize", "assign", "apply"]

import numpy as np
import pandas as pd
from toolz import curry


@curry
def groupby(cols, df):
    """Call a dataframe's `.groupby` method"""
    return df.groupby(cols)


@curry
def rows(query, df):
    """Select rows using a `.query` (str), slicerange (start,stop,step), or indices (list)"""
    if isinstance(query, str):
        return df.query(query).reset_index(drop=True)
    elif isinstance(query, (list, np.ndarray)):
        return df.iloc[query, :]
    elif isinstance(query, tuple):
        return df.iloc[slice(*query), :]


@curry
def cols(query, df):
    """Select columns using a `.query` (str), slicerange (start,stop,step), or indices (list). Uses `plydata.select`"""
    if isinstance(query, str):
        from plydata import select

        return select(df, query)
    elif isinstance(query, (list, np.ndarray)):
        return df.iloc[:, query]
    elif isinstance(query, tuple):
        return df.iloc[:, slice(*query)]


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
