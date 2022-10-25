"""
Common data operations and transformations often on pandas dataframes. This creates
**new dataframe methods** that can be called like this:  

`df.norm_by_group(grpcol='Class', valcol='Score')`

---
"""
__all__ = [
    "norm_by_group",
    "assert_balanced_groups",
    "assert_same_nunique",
    "select",
    "to_long",
    "to_wide",
]

import numpy as np
import pandas as pd
from functools import wraps
from typing import Union, List
from pandas.api.extensions import register_dataframe_accessor
from pandas.core.groupby.groupby import GroupBy
from utilz import filtercat, mapcat
from itertools import permutations
from math import factorial
from toolz import curry


# Register a function as a method attached to the Pandas DataFrame. Note: credit for
# this code goes entirely to `pandas_flavor`. Using the source here simply avoids an
# unecessary dependencies.
def _register_dataframe_method(method):
    def inner(*args, **kwargs):
        class AccessorMethod(object):
            def __init__(self, pandas_obj):
                self._obj = pandas_obj

            @wraps(method)
            def __call__(self, *args, **kwargs):
                return method(self._obj, *args, **kwargs)

        register_dataframe_accessor(method.__name__)(AccessorMethod)

        return method

    return inner()


@_register_dataframe_method
def norm_by_group(df, grpcol, valcols, center=True, scale=True, addcol=True):
    """
    Normalize values in one or more columns separately per group

    Args:
        df (pd.DataFrame): input dataframe
        grpcols (str): grouping col
        valcols (Union[str, List]): value cols
        center (bool, optional): mean center. Defaults to True.
        scale (bool, optional): divide by standard deviation. Defaults to True.
    """

    def _norm(dat, center, scale):
        if center:
            dat = dat - dat.mean()
        if scale:
            dat = dat / dat.std()
        return dat

    if isinstance(grpcol, List):
        raise NotImplementedError("Grouping by multiple columns is not supported")

    if not isinstance(valcols, List):
        valcols = [valcols]

    out = df.groupby(grpcol)[valcols].transform(_norm, center, scale)

    if addcol:
        if center and not scale:
            idx = "centered"
        elif scale and not center:
            idx = "scaled"
        elif center and scale:
            idx = "normed"

        assign_dict = {}
        for valcol, col in zip(valcols, out):
            assign_dict[f"{valcol}_{idx}_by_{grpcol}"] = col
        out = df.assign(**assign_dict)
    return out.squeeze()


@_register_dataframe_method
def assert_balanced_groups(df, grpcols: Union[str, List], size=None):
    """
    Check if each group of `grpcols` has the same dimensions

    Args:
        df (pd.DataFrame): input dataframe
        group_cols (str/List): column names to group on in dataframe
        shape (tuple/None, optional): optional group sizes to ensure
    """

    grouped = df.groupby(grpcols).size()
    size = grouped[0] if size is None else size
    if not np.all(grouped == size):
        raise AssertionError(f"Group sizes don't match!\n{grouped}")
    else:
        return True


@_register_dataframe_method
def assert_same_nunique(df, grpcols: Union[str, List], valcol: str, size=None):
    """
    Check if each group has the same number of unique values in `valcol`

    Args:
        df (pd.DataFrame): input dataframe
        valcol (str): column to check unique values in
        grpcols (str/list): column names to group on in dataframe, Default None
        shape (tuple/None, optional): optional sizes to ensure
    """

    grouped = df.groupby(grpcols)[valcol].nunique()
    size = grouped[0] if size is None else size
    if not np.all(grouped == size):
        raise AssertionError(f"Groups don't have same nunique values!\n{grouped}")
    else:
        return True


@_register_dataframe_method
def select(df, *args, **kwargs):
    """
    Select one ore more columns by name. Drop one or more columns by prepending '-' to
    the name. Rename columns using keyword arguments.

    Examples:

        >>> # Grab 2 columns
        >>> df.select('sepal_width', 'petal_width')


        >>> # Get all columns except one
        >>> df.select('-sepal_width')

        >>> # Grab a column and rename it
        >>> df.select(sepal_width='width')

    """
    # "Select as" functionality; get col and rename
    if kwargs:
        if args:
            raise ValueError(
                "mixing arguments and keyword arguments is not supported. If you want to filter columns and rename them, you should instead chain multiple calls to .select. For example: df.select('-sepal_length').select(petal_width='width', species='flower')"
            )
        cols = list(kwargs.keys())
        return df.filter(items=cols, axis="columns").rename(columns=kwargs)

    # Get col via name or exclude -name
    col_list = [*args]
    # Split columns to keep and drop based on '-' prefix
    drop, keep = filtercat("-", col_list, invert="split", assert_notempty=False)
    # Remove the prefix
    if len(drop):
        drop = mapcat(lambda col: col[1:], drop)
    if len(keep):
        return df.drop(columns=drop).filter(items=keep, axis="columns")
    return df.drop(columns=drop)


def _select(dfg, *args):
    """Same function for monkeypatching groupby objects"""

    # Force return of Series groupby by if single column requested
    if len(args) == 1:
        col = args[0]
        if not col.startswith("-"):
            return dfg[args[0]]
        cols = filtercat(col[1:], dfg.obj.columns, invert=True)
        # Incase we only have 2 cols and filter out 1 ensure series return type
        cols = cols[0] if len(cols) == 1 else cols
        return dfg[cols]

    # Support selecting via col or -col
    col_list = [*args]

    # Split columns to keep and drop based on '-' prefix
    drop, keep = filtercat("-", col_list, invert="split", assert_notempty=False)

    # Remove the prefix
    if len(drop):
        drop = mapcat(lambda col: col[1:], drop)

    # Add the grouping cols to the drop list
    drop += dfg.grouper.names
    cols = filtercat(drop, dfg.obj.columns, invert=True, assert_notempty=False)

    if len(keep):
        cols = filtercat(keep, cols)

    # Incase we filter down to 1 col ensure series return type
    cols = cols[0] if len(cols) == 1 else cols
    return dfg[cols]


# Monkeypatch groupby object with a .select method
GroupBy.select = _select


@_register_dataframe_method
def to_long(df, columns, into=("variable", "value"), add_unique_id=False):
    if not isinstance(columns, list):
        columns = [columns]
    id_vars = [col for col in df.columns if col not in columns]
    # Warning this is expensive
    if add_unique_id:
        num_unique_id_labels = df[id_vars].melt()["value"].nunique()
        nuniques = int(
            factorial(num_unique_id_labels)
            / factorial(num_unique_id_labels - len(id_vars))
        )
        # nuniques = len(list(permutations(stacked_id_vars, len(id_vars))))
        if nuniques < df.shape[0]:
            print("non-unique index")
            uniquified = (
                df.reset_index()
                .rename(columns={"index": "row_id"})
                .assign(
                    unique_id=lambda df: df[id_vars + ["row_id"]].apply(
                        lambda row: "_".join(row.values.astype(str)), axis=1
                    )
                )
            )
            return uniquified.melt(
                id_vars="unique_id",
                value_vars=columns or list(df.columns),
                var_name=into[0],
                value_name=into[1],
            )

    return df.melt(
        id_vars=[col for col in df.columns if col not in columns],
        value_vars=columns or list(df.columns),
        var_name=into[0],
        value_name=into[1],
    )


@_register_dataframe_method
def to_wide(df, column, by):

    index = [col for col in df.columns if col not in [column, by]]
    try:
        out = df.pivot(
            index=index,
            columns=column,
            values=by,
        ).reset_index()
        return out
    except ValueError as e:
        if "duplicate" in str(e):
            print(
                f"ERROR: It's not possible to infer what rows are unique from columns that make up the index: {index}. If you have multiple observations per index, then you should use .pivot_table and decide how to *aggregate* these observations. Otherwise .to_long() can create a unique index for with the add_unique_id = True"
            )
        raise e
