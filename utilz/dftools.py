"""
Common data operations and transformations often on pandas dataframes. This creates
**new dataframe methods** that can be called like this:  

`df.norm_by_group(grpcol='Class', valcol='Score')`

---
"""
__all__ = ["norm_by_group", "assert_balanced_groups", "assert_same_nunique"]

import numpy as np
from functools import wraps
from typing import Union, List
from pandas.api.extensions import register_dataframe_accessor
from pandas.core.groupby.groupby import GroupBy
from utilz import filtercat, mapcat


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
    if kwargs:
        if args:
            raise ValueError(
                "mixing arguments and keyword arguments is not supported. If you want to filter columns and rename them, you should instead chain multiple calls to .select. For example: df.select('-sepal_length').select(petal_width='width', species='flower')"
            )
        cols = list(kwargs.keys())
        return df.filter(items=cols, axis="columns").rename(columns=kwargs)
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
    return dfg[[*args]]


# Monkeypatch groupby object with a .select method
GroupBy.select = _select
