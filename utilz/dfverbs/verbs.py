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
    "concat",
    "merge",
    "join",
    "ngroups",
    "squeeze",
    "to_numpy",
    "to_list",
    "ngroups",
    "get_group",
    "reset_index",
    "split_groups",
    "assign",
]

import pandas as pd
from toolz import curry
from ..ops import do
from ..maps import filter


def _reset_index_helper(out, reset_index):
    if reset_index == "drop":
        return out.reset_index(drop=True)
    if reset_index == "reset":
        return out.reset_index()
    return out


@curry
def squeeze(*args, **kwargs):
    """Call a dataframe's `.squeeze` method"""

    def call(df):
        return df.squeeze(*args, **kwargs)

    return call


@curry
def to_numpy(*args, **kwargs):
    """Call a dataframe's `.to_numpy` method"""

    def call(df):
        return df.to_numpy(*args, **kwargs)

    return call


@curry
def to_list(*args, **kwargs):
    """Call a dataframe's `.to_list` method"""

    def call(df):
        return df.to_list(*args, **kwargs)

    return call


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
def concat(*args, **kwargs):
    """Call pd.concat"""
    return pd.concat(*args, **kwargs)


@curry
def merge(*args, **kwargs):
    """Call pd.concat"""
    return pd.merge(*args, **kwargs)


@curry
def join(*args, **kwargs):
    """Call pd.concat"""
    return pd.join(*args, **kwargs)


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
    or `_.summarize(weight_mean = lambda weight: weight.mean())` or `_.summarize(weight_mean = lambda df: df['weight].mean())`. To return output the
    same size as the input dataframe use `.mutate()` or `.transmute()` instead as
    either will *broadcast* values to the right size.
    """

    if isinstance(dfg, pd.core.groupby.generic.DataFrameGroupBy):
        out = None
        for k, v in kwargs.items():
            if isinstance(v, str):
                res = dfg.apply(lambda group: group.eval(v)).reset_index()
            elif callable(v):
                name = v.__code__.co_varnames
                if len(name) == 1:
                    if name[0] in ["df", "g", "group"]:
                        res = dfg.apply(v).reset_index()
                    else:
                        # Single column summarize
                        res = dfg.apply(lambda g: v(g[name[0]])).reset_index()
                else:
                    # Multi-column summarize
                    res = dfg.apply(lambda g: v(*[g[e] for e in name])).reset_index()
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
                name = v.__code__.co_varnames
                if len(name) == 1:
                    if name[0] == "df":
                        out[k] = v(dfg)
                    else:
                        # Single column summarize
                        out[k] = v(dfg[name[0]])
                else:
                    # multi-col summarize
                    cols = [dfg[e] for e in name]
                    out[k] = v(*cols)
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
     or `_.mutate(weight_centered = lambda weight: weight - weight.mean())` or `_.mutate(weight_centered = lambda df: df['weight].apply(lambda x: x -
     x.mean())`. To return output *smaller* than the input dataframe use `.summarize()` instead.
    """

    if isinstance(dfg, pd.core.groupby.generic.DataFrameGroupBy):
        prev = dfg.obj.copy()
        for _, (k, v) in enumerate(kwargs.items()):
            if isinstance(v, str):
                res = dfg.apply(lambda group: group.eval(v)).reset_index()
            elif callable(v):
                name = v.__code__.co_varnames
                if len(name) == 1:
                    # Normal assign where we pass in the entire dataframe to the calling
                    # function
                    if name[0] in ["df", "g", "group"]:
                        res = dfg.apply(v).reset_index()
                    else:
                        # Single column apply
                        res = dfg.apply(lambda g: v(g[name[0]])).reset_index()
                else:
                    # Multi-columm
                    res = dfg.apply(lambda g: v(*[g[e] for e in name])).reset_index()
            else:
                raise TypeError(
                    f"grouped dataframes cannot make direct assignments. You must pass in a str to be evaluated or a function but you passed in a type{v}"
                )

            # Calling an operation that returns df the same size as the original df,
            # like transform, e.g. 'A1 - A1.mean()'
            if res.shape[0] == prev.shape[0]:
                level_col_idx, level_col_name = [
                    (i, col)
                    for i, col in enumerate(res.columns)
                    if str(col).startswith("level_")
                ][0]

                res = res.rename(columns={res.columns[-1]: k})

                # Allow column overwriting
                if k in prev:
                    prev = prev.drop(columns=k).merge(
                        res.iloc[:, level_col_idx:],
                        left_index=True,
                        right_on=level_col_name,
                    )
                else:
                    # prev = prev.join(res[k])
                    prev = prev.merge(
                        res.iloc[:, level_col_idx:],
                        left_index=True,
                        right_on=level_col_name,
                    )
                prev = prev.drop(columns=level_col_name).reset_index(drop=True)
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
        out = dfg.copy()
        for k, v in kwargs.items():
            if isinstance(v, str):
                out = out.assign(**{k: dfg.eval(v)})
            elif callable(v):
                name = v.__code__.co_varnames
                if len(name) == 1:
                    # Normal assign where we pass in the entire dataframe to the calling
                    # function
                    if name[0] == "df":
                        out = out.assign(**{k: v})
                    else:
                        # Single column apply
                        out = out.assign(**{k: lambda df: v(df[name[0]])})
                else:
                    # Multi-columm
                    # get columns as list
                    cols = [dfg[e] for e in name]
                    out = out.assign(**{k: v(*cols)})
            else:
                # Normal assignment
                out = out.assign(**{k: v})

        return out


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
    cols = filter(list(orig.columns), list(out.columns), substr_match=False)
    out = out.drop(columns=cols)

    if out.shape[1] < 1:
        raise ValueError(
            "transmute does not support reassigning to an existing column. Give your new column(s) a different name(s) to extract"
        )
    else:
        return out


@curry
def query(q, **kwargs):
    """
    Call a dataframe object's `.query` method. Resets and drops index by
    default. Change this with `reset_index='drop'|'reset'|'none'`
    """
    reset_index = kwargs.pop("reset_index", "drop")

    def call(df):
        if isinstance(q, str):
            df = df.query(q, **kwargs)
        elif callable(q):
            name = q.__code__.co_varnames
            if len(name) == 1:
                if name[0] == "df":
                    df = df.loc[q]
                else:
                    df = df[q(df[name[0]])]
            else:
                df = df[q(*[df[e] for e in name])]

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
            out = _reset_index_helper(out, reset_index)
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
    the name. **Always returns a dataframe** even if there is just 1 column. Does not support renaming
    """

    def call(df):
        return do("select", df, *args)

    return call


@curry
def pivot_wider(*args, **kwargs):
    """
    Convert a pair of columns to multiple columns, e.g. `_.pivot_wider('condition', using='response')`

    Args:
        column (str): string name of column to "explode"
        using (str): string name of column who's values should be placed into the new columns
        drop_index (bool; optional): if a 'prev_index' col exists (usually created by
        make_index=True in pivot_longer) will drop it; Default True

    """

    def call(df):
        return df.pivot_wider(*args, **kwargs)

    return call


@curry
def pivot_longer(*args, **kwargs):
    """
    Convert a list of columns into 2 columns. Can pass a list of columsn to melt-down or
    `id_vars` to select everything else: e.g. `_.pivot_longer(['male', 'female'],
    into=('gender', 'response'))` or `_.pivot_longer(id_vars='SID', into=('gender','response'))`

    Args:
        columns (list or None): columns to melt; Defaults to None
        id_vars (list or None): columns to use as id variables; Default to None
        into (tuple, optional): cols to create Defaults to ("variable", "value").
        make_index (bool, optional): does a reset_index prior to melting and adds the
        index col to id_vars. Defaults to False.

    """

    def call(df):
        return df.pivot_longer(*args, **kwargs)

    return call


@curry
def split(*args, sep=" "):
    """
    Split values in single df column into multiple columns by separator, e.g.
    First-Last -> [First], [Last]. To split list elements use [] as the sep, e.g.
    [1,2,3] -> [1], [2], [3]

    Args:
        column (str): column to split
        into (list): new columns names to create
        sep (str, list): separator to split on. Use [] for list

    """

    col, into = args

    def call(df):
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

    return call


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


@curry
def reset_index(*args, **kwargs):
    """Call a dataframe's reset_index method"""

    def call(df):
        return df.reset_index(*args, **kwargs)

    return call


@curry
def ngroups(*args, **kwargs):
    def call(dfg):
        if isinstance(dfg, pd.core.groupby.generic.DataFrameGroupBy):
            return dfg.ngroups
        raise TypeError("ngroups only works on grouped dataframes")

    return call


@curry
def get_group(group):
    def call(dfg):
        if isinstance(dfg, pd.core.groupby.generic.DataFrameGroupBy):
            if isinstance(group, str):
                return dfg.get_group(group)
            elif isinstance(group, int):
                return dfg.get_group(list(dfg.groups.keys())[group])
        raise TypeError("get_group only works on grouped dataframes")

    return call


@curry
def split_groups():
    def call(dfg):
        if isinstance(dfg, pd.core.groupby.generic.DataFrameGroupBy):
            return dfg.split_groups()
        raise TypeError("split_groups only works on grouped dataframes")

    return call
