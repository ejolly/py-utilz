"""
Common data operations and transformations often on pandas dataframes. This creates
**new dataframe methods** that can be called like this:  

`df.norm_by_group(grpcol='Class', valcol='Score')`

---
"""
__all__ = ["norm_by_group", "assert_balanced_groups", "assert_same_nunique"]

import numpy as np
from functools import wraps
from typing import Union
from pandas.api.extensions import register_dataframe_accessor


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
def norm_by_group(df, grpcol, valcol, center=True, scale=True, addcol=True):
    """
    Normalize values in a column separately per group

    Args:
        df (pd.DataFrame): input dataframe
        grpcols (str): grouping col
        valcol (str): value col
        center (bool, optional): mean center. Defaults to True.
        scale (bool, optional): divide by standard deviation. Defaults to True.
    """

    def _norm(dat, center, scale):
        if center:
            dat = dat - dat.mean()
        if scale:
            dat = dat / dat.std()
        return dat

    if isinstance(grpcol, list):
        raise NotImplementedError("Grouping by multiple columns is not supported")

    out = df.groupby(grpcol)[valcol].transform(_norm, center, scale)

    if addcol:
        if center and not scale:
            idx = "centered"
        elif scale and not center:
            idx = "scaled"
        elif center and scale:
            idx = "normed"
        return df.assign(**{f"{valcol}_{idx}_by_{grpcol}": out})
    else:
        return out


@_register_dataframe_method
def assert_balanced_groups(df, grpcols: Union[str, list], size=None):
    """
    Check if each group of `grpcols` has the same dimensions

    Args:
        df (pd.DataFrame): input dataframe
        group_cols (str/list): column names to group on in dataframe
        shape (tuple/None, optional): optional group sizes to ensure
    """

    grouped = df.groupby(grpcols).size()
    size = grouped[0] if size is None else size
    if not np.all(grouped == size):
        raise AssertionError(f"Group sizes don't match!\n{grouped}")
    else:
        return True


@_register_dataframe_method
def assert_same_nunique(df, grpcols: Union[str, list], valcol: str, size=None):
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
