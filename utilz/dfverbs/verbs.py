"""
dplyr like *verbs* for working with pandas dataframes.

"""

__all__ = [
    "mutate",
    "transmute",
    "summarize",
    "query",
    "sort",
    "groupby",
    "to_csv",
    "read_csv",
    "apply",
    "rename",
    "head",
    "tail",
    "drop",
    "select",
    "pivot_longer",
    "pivot_wider",
    "split",
    "astype",
    "replace",
    "fillna",
    "splitquery",
    "call",
]

import pandas as pd
from toolz import curry
from ..ops import do, filtercat


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
def read_csv(*args, **kwargs):
    """Call pd.read_csv"""
    return pd.read_csv(*args, **kwargs)


@curry
def to_csv(path, df, index=False):
    """Call a dataframe's `.to_csv(index=False)` method"""
    if not str(path).endswith(".csv"):
        path = f"{path}.csv"
    df.to_csv(f"{path}", index=index)
    return df


@curry
def summarize(dfg, **kwargs):
    """
    Create new columns based on existing columns in a dataframe but return a
    **smaller** dataframe than the original. Works with the output of `groupby` as well:

    Just like `.mutate()/.transmute()`, input should be kwargs organized like
    `new_column = str| function`. Such as: `_.summarize(weight_mean ='weight.mean()')`
    or `_.summarize(weight_mean = lambda df: df['weight].mean())`. To return output the
    same size as the input dataframe use `.mutate()` or `.transmute()` instead as
    either will *broadcast* values to the right size.
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
            else:
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
            f"summarize expected previous step to be a DataFrame or GroupBy, but received a {type(dfg)}. If you used select(), you should instead select the column in the expression or function passed to summarize(new_col='old_col.mean()'). If you intended to run an expression summarize takes kwargs organized like: new_colname = str | func. This differs from agg in pandas which expects a column name and expression!"
        )


@curry
def assign(**kwargs):
    """Call a dataframe object's `.assign` method"""

    def call(df):
        out = df.assign(**kwargs)
        return out

    return call


@curry
def mutate(dfg, **kwargs):
    """
    Creates a new column(s) in a DataFrame based on a function of existing columns in
    the DataFrame. Always returns a dataframe the same size as the original. For groupby
    inputs, **the result is always ungrouped.**

    Just like `.summarize()`, input should be kwargs organized like `new_column = str|
    function`. Such as: `_.mutate(weight_centered ='weight - weight.mean()')`
     or `_.mutate(weight_centered_ = lambda df: df['weight].apply(lambda x: x -
     x.mean())`. To return output *smaller* than the input dataframe use `.summarize()` instead.
    """

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
        if any(map(lambda e: isinstance(e, str), kwargs.values())):
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
def transmute(dfg, **kwargs):
    """Just like `.mutate()`, but only returns the newly created columns."""
    if isinstance(
        dfg,
        (
            pd.core.groupby.generic.DataFrameGroupBy,
            pd.core.groupby.generic.SeriesGroupBy,
        ),
    ):
        orig = dfg.obj
    else:
        orig = dfg
    out = mutate(dfg, **kwargs)
    cols = filtercat(list(orig.columns), list(out.columns), substr_match=False)
    out = out.drop(columns=cols)

    if out.shape[1] < 1:
        raise ValueError(
            "transmute does not support reassigning to an existing column. Give your new column(s) a different name(s) to extract"
        )
    else:
        return out


@curry
def query(*queries, **kwargs):
    """
    Call a dataframe object's `.query` method. Resets and drops index by
    default. Change this with `reset_index='drop'|'reset'|'none'`
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
    """Call a dataframe or groupby object's `.apply` method
    For groupbed dataframes, resets and drops index by default. Change this with `reset_index='drop'|'reset'|'none'`
    """

    reset_index = kwargs.pop("reset_index", "drop")

    def call(df):
        out = df.apply(*args, **kwargs)
        if isinstance(df, pd.core.groupby.generic.DataFrameGroupBy):
            out = (_reset_index_helper(out, reset_index),)
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
    """Call a dataframe's `.drop(axis=1)` method. Column names should be passed as
    multiple args like `.select()`, e.g. `_.drop('height', 'weight')`"""

    def call(df):
        return do("drop", df, [*args], axis=1)

    return call


@curry
def select(*args):
    """
    Select one or more columns by name. Drop one or more columns by prepending '-' to
    the name. **Always returns a dataframe** even if there is just 1 column. Does not support renaming"""

    def call(df):
        return do("select", df, *args)

    return call


@curry
def pivot_wider(*args, **kwargs):
    """
    Convert a pair of columns to multiple columns, e.g. `_.to_wide('condition', using='response')`

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
def pivot_longer(*args, **kwargs):
    """
    Convert a list of columns into 2 columns. Can pass a list of columsn to melt-down or
    `id_vars` to select everything else: e.g. `_.to_long(['male', 'female'],
    into=('gender', 'response'))` or `_.to_long(id_vars='SID', into=('gender','response'))`

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
    """Cast one ore more columns to a type. Like `.rename()` you can either input a single tuple to cast 1
    column or a dict to cast multiple"""
    if isinstance(cols, tuple):
        cols = {cols[0]: cols[1]}
    return df.astype(cols)


@curry
def sort(*args, **kwargs):
    """Sort df by one or more columns passed as args. Ignores index by default by you
    can change that with `ignore_index=False`."""
    ignore_index = kwargs.pop("ignore_index", True)

    def call(df):
        return df.sort_values(by=list(args), ignore_index=ignore_index, **kwargs)

    return call


@curry
def call(*args, **kwargs):
    """Call an arbitrary method or function on an object, e.g. `pipe(df,
    _.call('mean'))` would call `df.mean()`"""

    def _call(df):
        method_name = args[0]
        func = getattr(df, method_name, None)
        if func is not None:
            return func(*args[1:], **kwargs)
        else:
            raise AttributeError(f"{type(df)} does not have a {method_name} method")

    return _call


@curry
def splitquery(query, **kwargs):
    """
    Call a dataframe or groupby object's `.query` method and return 2 dataframes one
    where containing results where the query is true and its inverse.
    Resets and drops index by default. Change this with `reset_index='drop'|'reset'|'none'`
    """
    reset_index = kwargs.pop("reset_index", "drop")

    def call(df):
        if isinstance(query, str):
            df_yes = df.query(query, **kwargs)
            df_no = df.query(f"not ({query})", **kwargs)
        elif callable(query):
            df_yes = df.loc[query]
            df_no = df.loc[~(query)]

        return (
            _reset_index_helper(df_yes, reset_index),
            _reset_index_helper(df_no, reset_index),
        )

    return call


@curry
def fillna(*args, **kwargs):
    """Call a dataframe's fillna method"""

    def call(df):
        return df.fillna(*args, **kwargs)

    return call


@curry
def replace(*args, **kwargs):
    """Call a dataframe's replace method"""

    def call(df):
        return df.replace(*args, **kwargs)

    return call
