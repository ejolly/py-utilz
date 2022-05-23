"""
Common data operations and transformations often on pandas dataframes

---
"""
__all__ = ["norm_by_group"]

from functools import wraps
from typing import Union
from pandas.api.extensions import register_dataframe_accessor


def _register_dataframe_method(method):
    """
    Register a function as a method attached to the Pandas DataFrame. Note: credit for
    this code goes entirely to `pandas_flavor`. Using the source here simply avoids an
    unecessary dependencies.
    """

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


# TODO: Test me
def same_shape(group_cols: Union[str, list], shape=None):
    """
    Check if each group of `group_col` has the same dimensions after running a function on a dataframe

    Args:
        group_cols (str/list): column names to group on in dataframe
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            df = func(*args, **kwargs)
            grouped = df.groupby(group_cols).size()
            if shape is None:
                if not grouped.sum() % grouped.shape[0] == 0:
                    raise AssertionError("Groups dont have the same shape", grouped)
            else:
                if not all(grouped == shape):
                    raise AssertionError(
                        f"All groups dont match shape {shape}", grouped
                    )
            return df

        return wrapper

    return decorator


# TODO: write me
# def same_nunique(func: callable, val_col: str, group_col: str):
#     """
#     Check if each group of `group_col` has the same number of unique values of `val_col` after running a function on a dataframe

#     Args:
#         func (callable): a function that operates on a dataframe
#         val_col (str): column name to check for unique values in dataframe
#         group_col (str): column name to group on in dataframe
#     """

#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         pass

#     return wrapper
