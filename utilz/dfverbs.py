import pandas as pd
from toolz import curry

__all__ = ["assign", "apply", "groupby", "head", "tail"]


@curry
def assign(df, *args, **kwargs):
    """Call a dataframe's assign method. Also works with groupby objects"""

    if isinstance(df, pd.core.groupby.generic.DataFrameGroupBy):
        prev = df.filter(lambda _: True).reset_index()
        for _, (k, v) in enumerate(kwargs.items()):
            res = df.apply(lambda group: group.eval(v)).reset_index()
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
        return df.assign(*args, **kwargs)


@curry
def apply(func, df, drop=True):
    """Call a dataframe or groupby object's `.apply` method followed by a reset_index"""
    out = df.apply(func)
    return out.reset_index(drop=drop)


@curry
def groupby(groups, df):
    """Call a dataframe's groupby method"""
    return df.groupby(groups)


@curry
def head(df, **kwargs):
    """Call a dataframe's head method"""
    return df.head(**kwargs)


@curry
def tail(df, **kwargs):
    """Call a dataframe's tail method"""
    return df.tail(**kwargs)
