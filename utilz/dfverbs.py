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
    """Rename one ore more columns. Can either input a single tuple to rename 1 column
    or a dict to rename multiple"""
    if isinstance(cols, tuple):
        cols = {cols[0]: cols[1]}
    return df.rename(columns=cols)


@curry
def to_csv(path, df):
    """Call a dataframe's `.to_csv(index=False)` method"""
    df.to_csv(f"{path}.csv", index=False)
    return df


@curry
def summarize(dfg, **kwargs):
    """
    Creates a new column(s) in a DataFrame based on a function of existing columns in the DataFrame. Uses `plydata.define/mutate` unless the input is a grouped DataFrame
    in which case it falls back to pandas methods because `plydata` can only handle
    grouped inputs resulting from its own (slow) `group_by` function
    """

    if isinstance(dfg, pd.core.groupby.generic.DataFrameGroupBy):
        out = None
        for k, v in kwargs.items():
            if isinstance(v, str):
                res = dfg.apply(lambda group: group.eval(v)).reset_index()
            elif callable(v):
                res = dfg.apply(v).reset_index()
            else:
                raise TypeError(
                    f"summarize expects input kwargs organized like: new_colname = str | func, but receive type: {type(v)}"
                )
            res = res.rename(columns={res.columns[-1]: k})
            if not res.shape[0] < dfg.obj.shape[0]:
                raise ValueError(
                    "functions and expressions received by summarize should return a scalar output. If you want to broadcast this value over the entire dataframe use assign() instead."
                )
            if out is None:
                out = res
            out = out.drop(columns=k, errors="ignore").merge(
                res, on=res.columns[:-1].to_list()
            )
        return out
    elif isinstance(dfg, pd.DataFrame):
        out = dict()
        for k, v in kwargs.items():
            if isinstance(v, str):
                out[k] = dfg.eval(v)
            elif callable(v):
                out[k] = v(dfg)
            else:
                raise TypeError(
                    f"summarized expects input kwargs organized like: new_colname = str | func, but receive type: {type(v)}"
                )

        return pd.DataFrame(out, index=[0])
    else:
        raise TypeError(
            f"summarize expected previous step to be a DataFrame or GroupBy, but received a {type(dfg)}. If you used select(), you should instead select the column in the expression or function passed to summarize(). If you intended to run an expression summarize taks wargs organized like: new_colname = str | func. This differs from agg in pandas which expects an column name and expression!"
        )


# @curry
# def summarize(*args, **kwargs):
#     """
#     Summarize the output of one or more columns. Accepts grouped or ungrouped
#     dataframe. If following a select() just pass funcs directly:
#     pipe(data, select('col'), summarize('mean','std'))

#     Otherwise you can pass in kwargs to select and run funcs for diff columns:
#     pipe(data, summarize(col=['mean','std'], col2=['mode']))
#     """

#     tidy = kwargs.pop("tidy", True)

#     def call(df):
#         if len(args) == 1 and isinstance(args[0], dict):
#             out = df.agg(args[0])
#         elif len(args) == 0:
#             out = df.agg(kwargs)
#         else:
#             out = df.agg((args))
#         if tidy:
#             # groupby
#             if isinstance(
#                 df,
#                 (
#                     pd.core.groupby.generic.DataFrameGroupBy,
#                     pd.core.groupby.generic.SeriesGroupBy,
#                 ),
#             ):

#                 num_grps = len(out.index.names)
#                 unstacker = list(range(num_grps))
#                 breakpoint()
#                 out = out.unstack(unstacker).reset_index()

#                 # groupby(), select(), summarize()
#                 # 1, 1, 1
#                 # = groupby series, 1 level of stacking
#                 # = level_0 is stat

#                 # 1, 1, 2
#                 # = groupby series, 1 level of stacking
#                 # = level_0 is stat

#                 # 1, 2, 1
#                 # = groupby dataframe, 1 level of stacking
#                 # = level_0 is column name
#                 # = level_1 i stat

#                 # 1, 2, 2
#                 # = groupby dataframe, 1 level of stacking
#                 # = level_0 is column name
#                 # = level_1 i stat

#                 # 2, 1, 1
#                 # = groupby series, 2 levels of stacking
#                 # = level_0 is stat

#                 # 2, 1, 2
#                 # = groupby series, 2 levels of stacking
#                 # = level_0 is stat

#                 # 2, 2, 2
#                 # = groupby dataframe, 2 levels of stacking
#                 # = level_0 is column name
#                 # = level_1 is stat
#                 # need dropna if groups are nested

#                 if isinstance(df, pd.core.groupby.generic.DataFrameGroupBy):
#                     out = out.rename(
#                         columns={"level_0": "column", "level_1": "stat", 0: "value"}
#                     )
#                     colnames = ["column", "stat", "value"]
#                 else:
#                     out = out.rename(columns={"level_0": "stat", 0: "value"})
#                     colnames = ["stat", "value"]

#                 if len(args) and isinstance(args[0], str):
#                     out = out.assign(
#                         stat=np.repeat(args, int(out.shape[0] / len(args)))
#                     )

#                 # Rearrange cols
#                 cols = list(out.columns)
#                 created, groups = filtercat(colnames, cols, invert="split")
#                 new_order = groups + created
#                 out = (
#                     out[new_order]
#                     .dropna()
#                     .sort_values(by=groups)
#                     .reset_index(drop=True)
#                 )
#             else:
#                 # non-grouped
#                 out = (
#                     out.reset_index()
#                     .rename(columns={"index": "stat"})
#                     .melt(id_vars="stat", var_name="column")
#                 )
#                 out = out[["column", "stat", "value"]].dropna().reset_index(drop=True)

#             # Drop the column column if it only has one value, e.g. if we did a
#             # summarize on a long-form and provided kwarg to summarize
#             if "column" in out and out["column"].nunique() == 1:
#                 out = out.drop(columns="column")
#             return out

#         else:
#             return out

#     return call


@curry
def assign(dfg, **kwargs):
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
            return do("assign", dfg, **kwargs)


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
def apply(*args, **kwargs):
    """Call a dataframe or groupby object's `.apply` method"""

    def call(df):
        out = df.apply(*args, **kwargs)
        if isinstance(df, pd.core.groupby.generic.DataFrameGroupBy):
            out = out.reset_index()
        return out

    return call


@curry
def head(*args, **kwargs):
    """Call dataframe's `.head()` method"""

    def call(df):
        return df.head(*args, **kwargs)

    return call


@curry
def tail(*args, **kwargs):
    """Call dataframe's `.tail()` method"""

    def call(df):
        return df.tail(*args, **kwargs)

    return call


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

    return call


@curry
def to_wide(*args, **kwargs):
    """
    Convert a pair of columns to multiple columns

    Args:
        column (str): string name of column to "explode"
        using (str): string name of column who's values should be placed into the new columns
        drop_index (bool; optional): if a 'prev_index' col exists (usually created by
        make_index=True in to_long) will drop it; Default True

    """

    def call(df):
        return df.to_wide(*args, **kwargs)

    return call


@curry
def to_long(*args, **kwargs):
    """
    Convert a list of columns into 2 columns. Does not support renaming.

    Args:
        columns (list or None): columns to melt; Defaults to None
        id_vars (list or None): columns to use as id variables; Default to None
        into (tuple, optional): cols to create Defaults to ("variable", "value").
        make_index (bool, optional): does a reset_index prior to melting and adds the
        index col to id_vars. Defaults to False.

    """

    def call(df):
        return df.to_long(*args, **kwargs)

    return call


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
def astype(cols, df):
    """Cast one ore more columns to a type. Can either input a single tuple to cast 1
    column or a dict to cast multiple"""
    if isinstance(cols, tuple):
        cols = {cols[0]: cols[1]}
    return df.astype(cols)
