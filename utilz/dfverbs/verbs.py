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
import polars as pl
from toolz import curry
from ..ops import do
from ..maps import filter
from .polars_utils import is_polars_object, ensure_polars_expr, polars_curry


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
    """Call a dataframe's `.groupby` method. Works with both pandas and polars DataFrames."""

    def call(df):
        if is_polars_object(df):
            # Polars uses group_by instead of groupby
            return df.group_by(*args)
        else:
            return do("groupby", df, [*args])

    return call


@curry
def rename(cols, df):
    """Rename one ore more columns. Can either input a single tuple to rename 1 column
    or a dict to rename multiple. Works with both pandas and polars DataFrames."""
    if isinstance(cols, tuple):
        cols = {cols[0]: cols[1]}
    
    if is_polars_object(df):
        return df.rename(cols)
    else:
        return df.rename(columns=cols)


@curry
def read_csv(*args, **kwargs):
    """Call pd.read_csv or pl.read_csv based on use_polars parameter"""
    use_polars = kwargs.pop('use_polars', False)
    if use_polars:
        return pl.read_csv(*args, **kwargs)
    return pd.read_csv(*args, **kwargs)


@curry
def concat(*args, **kwargs):
    """Call pd.concat or pl.concat. Works with both pandas and polars DataFrames."""
    # Check if first dataframe is polars
    dfs = args[0] if args else kwargs.get('objs', [])
    if dfs and is_polars_object(dfs[0]):
        # Polars concat
        how = kwargs.pop('how', 'vertical')
        return pl.concat(dfs, how=how, **kwargs)
    return pd.concat(*args, **kwargs)


@curry
def merge(*args, **kwargs):
    """Call pd.merge or polars join. Works with both pandas and polars DataFrames."""
    # For merge, we need at least one dataframe
    if args:
        left = args[0]
        right = args[1] if len(args) > 1 else kwargs.get('right')
        
        if is_polars_object(left):
            # Extract merge parameters
            on = kwargs.get('on', None)
            how = kwargs.get('how', 'inner')
            left_on = kwargs.get('left_on', None)
            right_on = kwargs.get('right_on', None)
            
            # Polars join - use either 'on' or 'left_on'/'right_on'
            if on is not None:
                return left.join(right, on=on, how=how)
            elif left_on is not None or right_on is not None:
                return left.join(right, left_on=left_on, right_on=right_on, how=how)
            else:
                # Try to join on common columns
                return left.join(right, how=how)
    
    return pd.merge(*args, **kwargs)


@curry
def join(*args, **kwargs):
    """Call dataframe join method. Works with both pandas and polars DataFrames."""
    def call(df):
        if is_polars_object(df):
            # For polars, join is a method on the dataframe
            other = args[0] if args else kwargs.get('other')
            on = kwargs.get('on', None)
            how = kwargs.get('how', 'left')
            return df.join(other, on=on, how=how)
        else:
            return df.join(*args, **kwargs)
    
    return call


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
    
    Works with both pandas and polars DataFrames.
    """
    
    # Handle polars DataFrames and GroupBy objects
    if is_polars_object(dfg) or (hasattr(dfg, '__class__') and 'polars' in str(dfg.__class__.__module__)):
        return _summarize_polars(dfg, **kwargs)

    # Handle pandas grouped DataFrames
    elif isinstance(dfg, pd.core.groupby.generic.DataFrameGroupBy):
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


def _summarize_polars(dfg, **kwargs):
    """
    Helper function to handle summarize operations on polars DataFrames.
    """
    # Convert to list of expressions for aggregation
    expressions = []
    
    for k, v in kwargs.items():
        if isinstance(v, str):
            # Handle string expressions
            # Common aggregations: mean(), sum(), min(), max(), count()
            try:
                # Try to parse as aggregation expression
                if '.' in v:
                    col_name, agg_func = v.rsplit('.', 1)
                    agg_func = agg_func.rstrip('()')
                    
                    if agg_func == 'mean':
                        expr = pl.col(col_name).mean().alias(k)
                    elif agg_func == 'sum':
                        expr = pl.col(col_name).sum().alias(k)
                    elif agg_func == 'min':
                        expr = pl.col(col_name).min().alias(k)
                    elif agg_func == 'max':
                        expr = pl.col(col_name).max().alias(k)
                    elif agg_func == 'count':
                        expr = pl.col(col_name).count().alias(k)
                    elif agg_func == 'std':
                        expr = pl.col(col_name).std().alias(k)
                    elif agg_func == 'var':
                        expr = pl.col(col_name).var().alias(k)
                    else:
                        # Try generic sql expression
                        expr = pl.sql_expr(v).alias(k)
                else:
                    # Try as SQL expression
                    expr = pl.sql_expr(v).alias(k)
            except:
                # Fall back to treating as column name
                expr = pl.col(v).alias(k)
        elif isinstance(v, pl.Expr):
            # Already a polars expression
            expr = v.alias(k)
        elif callable(v):
            # For now, raise an error - lambda support needs more work
            raise NotImplementedError("Lambda expressions not yet supported for polars DataFrames in summarize")
        else:
            # Literal value
            expr = pl.lit(v).alias(k)
            
        expressions.append(expr)
    
    # Check if dfg is a GroupBy object
    if hasattr(dfg, 'agg') and hasattr(dfg, '__class__') and 'GroupBy' in str(dfg.__class__.__name__):
        # It's a grouped dataframe, use agg
        return dfg.agg(expressions)
    else:
        # It's a regular dataframe, use select to create single-row output
        return dfg.select(expressions).head(1)


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
    
    Works with both pandas and polars DataFrames.
    """
    
    # Handle polars DataFrames
    if is_polars_object(dfg):
        return _mutate_polars(dfg, **kwargs)
    
    # Handle pandas grouped DataFrames
    elif isinstance(dfg, pd.core.groupby.generic.DataFrameGroupBy):
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
    
    # Handle regular pandas DataFrames
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


def _mutate_polars(df, **kwargs):
    """
    Helper function to handle mutate operations on polars DataFrames.
    """
    # Convert to list of expressions for with_columns
    expressions = []
    
    for k, v in kwargs.items():
        if isinstance(v, str):
            # Handle string expressions using pl.sql_expr for SQL-like syntax
            try:
                expr = pl.sql_expr(v).alias(k)
            except:
                # Fall back to treating as column name
                expr = pl.col(v).alias(k)
        elif isinstance(v, pl.Expr):
            # Already a polars expression
            expr = v.alias(k)
        elif callable(v):
            # For now, raise an error - lambda support needs more work
            raise NotImplementedError("Lambda expressions not yet supported for polars DataFrames")
        else:
            # Literal value
            expr = pl.lit(v).alias(k)
            
        expressions.append(expr)
    
    return df.with_columns(expressions)


@curry
def transmute(dfg, **kwargs):
    """Just like `.mutate()`, but only returns the newly created columns. Works with both pandas and polars DataFrames."""
    
    # Handle polars DataFrames
    if is_polars_object(dfg):
        return _transmute_polars(dfg, **kwargs)
    
    # Handle pandas DataFrames (original logic)
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


def _transmute_polars(df, **kwargs):
    """
    Helper function to handle transmute operations on polars DataFrames.
    Returns only the newly created columns.
    """
    # Convert to list of expressions for select
    expressions = []
    
    for k, v in kwargs.items():
        if isinstance(v, str):
            # Handle string expressions using pl.sql_expr for SQL-like syntax
            try:
                expr = pl.sql_expr(v).alias(k)
            except:
                # Fall back to treating as column name
                expr = pl.col(v).alias(k)
        elif isinstance(v, pl.Expr):
            # Already a polars expression
            expr = v.alias(k)
        elif callable(v):
            # For now, raise an error - lambda support needs more work
            raise NotImplementedError("Lambda expressions not yet supported for polars DataFrames")
        else:
            # Literal value
            expr = pl.lit(v).alias(k)
            
        expressions.append(expr)
    
    return df.select(expressions)


@curry
def query(q, **kwargs):
    """
    Call a dataframe object's `.query` method. Resets and drops index by
    default. Change this with `reset_index='drop'|'reset'|'none'`. 
    Works with both pandas and polars DataFrames.
    """
    reset_index = kwargs.pop("reset_index", "drop")

    def call(df):
        if is_polars_object(df):
            # Handle polars DataFrames
            if isinstance(q, str):
                # Use sql_expr for string expressions
                try:
                    filter_expr = pl.sql_expr(q)
                except:
                    # Fall back to simple column comparison if sql_expr fails
                    filter_expr = pl.col(q)
                df = df.filter(filter_expr)
            elif callable(q):
                # For now, raise an error - lambda support needs more work
                raise NotImplementedError("Lambda expressions not yet supported for polars DataFrames in query")
            return df
        else:
            # Handle pandas DataFrames (original logic)
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
    the name. **Always returns a dataframe** even if there is just 1 column. Does not support renaming.
    Works with both pandas and polars DataFrames.
    """

    def call(df):
        if is_polars_object(df):
            return _select_polars(df, *args)
        else:
            return do("select", df, *args)

    return call


def _select_polars(df, *args):
    """
    Helper function to handle select operations on polars DataFrames.
    """
    columns_to_select = []
    columns_to_drop = []
    
    for arg in args:
        if isinstance(arg, str):
            if arg.startswith('-'):
                # Drop column (remove the '-' prefix)
                columns_to_drop.append(arg[1:])
            else:
                # Select column
                columns_to_select.append(arg)
        else:
            # Assume it's a column name
            columns_to_select.append(str(arg))
    
    # If we have columns to drop, drop them first
    if columns_to_drop:
        df = df.drop(columns_to_drop)
    
    # If we have columns to select, select only those
    if columns_to_select:
        df = df.select(columns_to_select)
    
    return df


@curry
def pivot_wider(*args, **kwargs):
    """
    Convert a pair of columns to multiple columns, e.g. `_.pivot_wider('condition', using='response')`.
    Works with both pandas and polars DataFrames.

    Args:
        column (str): string name of column to "explode"
        using (str): string name of column who's values should be placed into the new columns
        drop_index (bool; optional): if a 'prev_index' col exists (usually created by
        make_index=True in pivot_longer) will drop it; Default True

    """

    def call(df):
        if is_polars_object(df):
            # Extract args
            if len(args) >= 2:
                column, using = args[0], kwargs.get('using', args[1])
            else:
                column = args[0] if args else kwargs.get('column')
                using = kwargs.get('using')
            
            # Get index columns (all columns except the two being pivoted)
            index_cols = [col for col in df.columns if col not in [column, using]]
            
            # Use polars pivot with new parameter names
            return df.pivot(
                values=using,
                index=index_cols,
                on=column,  # Changed from 'columns' to 'on'
                aggregate_function="first"  # Use first value if duplicates
            )
        else:
            return df.pivot_wider(*args, **kwargs)

    return call


@curry
def pivot_longer(*args, **kwargs):
    """
    Convert a list of columns into 2 columns. Can pass a list of columsn to melt-down or
    `id_vars` to select everything else: e.g. `_.pivot_longer(['male', 'female'],
    into=('gender', 'response'))` or `_.pivot_longer(id_vars='SID', into=('gender','response'))`.
    Works with both pandas and polars DataFrames.

    Args:
        columns (list or None): columns to melt; Defaults to None
        id_vars (list or None): columns to use as id variables; Default to None
        into (tuple, optional): cols to create Defaults to ("variable", "value").
        make_index (bool, optional): does a reset_index prior to melting and adds the
        index col to id_vars. Defaults to False.

    """

    def call(df):
        if is_polars_object(df):
            # Extract arguments
            columns = args[0] if args else kwargs.get('columns', None)
            id_vars = kwargs.get('id_vars', None)
            into = kwargs.get('into', ('variable', 'value'))
            
            # If columns is provided, infer id_vars
            if columns is not None:
                if id_vars is None:
                    id_vars = [col for col in df.columns if col not in columns]
                value_vars = columns
            # If id_vars is provided, infer columns
            elif id_vars is not None:
                value_vars = [col for col in df.columns if col not in id_vars]
            else:
                # Neither provided, melt all columns
                id_vars = []
                value_vars = df.columns
            
            # Use polars unpivot (new name for melt)
            return df.unpivot(
                index=id_vars,  # Changed from 'id_vars' to 'index'
                on=value_vars,  # Changed from 'value_vars' to 'on'
                variable_name=into[0],
                value_name=into[1]
            )
        else:
            return df.pivot_longer(*args, **kwargs)

    return call


@curry
def split(*args, sep=" "):
    """
    Split values in single df column into multiple columns by separator, e.g.
    First-Last -> [First], [Last]. To split list elements use [] as the sep, e.g.
    [1,2,3] -> [1], [2], [3]. Works with both pandas and polars DataFrames.

    Args:
        column (str): column to split
        into (list): new columns names to create
        sep (str, list): separator to split on. Use [] for list

    """

    col, into = args

    def call(df):
        if is_polars_object(df):
            if isinstance(sep, str):
                # Use polars string split
                split_expr = pl.col(col).str.split(sep)
                # Create individual columns from the split
                new_cols = []
                for i, new_col in enumerate(into):
                    new_cols.append(
                        split_expr.list.get(i).alias(new_col)
                    )
                # Drop original column and add new ones
                return df.with_columns(new_cols).drop(col)
            elif isinstance(sep, list):
                # Handle list splitting
                # Extract list elements into separate columns
                new_cols = []
                for i, new_col in enumerate(into):
                    new_cols.append(
                        pl.col(col).list.get(i).alias(new_col)
                    )
                return df.with_columns(new_cols).drop(col)
        else:
            # Original pandas implementation
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
    column or a dict to cast multiple. Works with both pandas and polars DataFrames."""
    if isinstance(cols, tuple):
        cols = {cols[0]: cols[1]}
    
    if is_polars_object(df):
        # Polars uses cast() instead of astype()
        # Convert pandas dtype strings to polars dtypes
        cast_exprs = []
        for col_name, dtype in cols.items():
            if isinstance(dtype, str):
                # Convert common pandas dtype strings to polars
                dtype_map = {
                    'int64': pl.Int64,
                    'int32': pl.Int32,
                    'float64': pl.Float64,
                    'float32': pl.Float32,
                    'object': pl.Utf8,
                    'string': pl.Utf8,
                    'str': pl.Utf8,
                    'bool': pl.Boolean,
                    'datetime64[ns]': pl.Datetime,
                    'category': pl.Categorical,
                }
                pl_dtype = dtype_map.get(dtype, pl.Utf8)
            else:
                pl_dtype = dtype
            cast_exprs.append(pl.col(col_name).cast(pl_dtype))
        
        return df.with_columns(cast_exprs)
    else:
        return df.astype(cols)


@curry
def sort(*args, **kwargs):
    """Sort df by one or more columns passed as args. Ignores index by default but you
    can change that with `ignore_index=False`. Works with both pandas and polars DataFrames."""
    ignore_index = kwargs.pop("ignore_index", True)
    ascending = kwargs.pop("ascending", True)

    def call(df):
        if is_polars_object(df):
            # Polars uses sort() instead of sort_values()
            # Handle descending parameter (polars uses descending instead of ascending)
            if isinstance(ascending, bool):
                descending = not ascending
            else:
                # If ascending is a list, convert to descending list
                descending = [not asc for asc in ascending]
            return df.sort(list(args), descending=descending, **kwargs)
        else:
            return df.sort_values(by=list(args), ignore_index=ignore_index, ascending=ascending, **kwargs)

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
    """Call a dataframe's fillna method. Works with both pandas and polars DataFrames."""

    def call(df):
        if is_polars_object(df):
            # Polars uses fill_null() instead of fillna()
            value = args[0] if args else kwargs.get('value', None)
            return df.fill_null(value)
        else:
            return df.fillna(*args, **kwargs)

    return call


@curry
def replace(*args, **kwargs):
    """Call a dataframe's replace method. Works with both pandas and polars DataFrames."""

    def call(df):
        if is_polars_object(df):
            # Polars replace works differently - it requires old and new values
            if len(args) >= 2:
                old, new = args[0], args[1]
            else:
                old = kwargs.get('to_replace', kwargs.get('old', None))
                new = kwargs.get('value', kwargs.get('new', None))
            
            if old is not None and new is not None:
                # Need to handle column-specific replacements
                # For now, apply to all string columns
                expr_list = []
                for col in df.columns:
                    if df[col].dtype == pl.Utf8:
                        expr_list.append(pl.col(col).replace(old, new))
                    else:
                        expr_list.append(pl.col(col))
                return df.select(expr_list)
            else:
                raise ValueError("replace requires both old and new values for polars DataFrames")
        else:
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


# Make every verb usable with the unix-style `|` pipe operator, e.g.
# `df | mutate(c="a + b") | summarize(...)`, in addition to `pipe(df, ...)`.
from ..pipes import _pipeify  # noqa: E402

for _name in dict.fromkeys(__all__):
    globals()[_name] = _pipeify(globals()[_name])
