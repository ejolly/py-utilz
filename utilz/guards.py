"""
Custom guards for defensive data analysis compatible with [bulwark](https://bulwark.readthedocs.io/en/latest/index.html).

Intended usage is as Python decorators:

```
from utilz.guards import log_df

@log_df
def myfunc(df):
    do some stuff...
```

---
"""
__all__ = [
    "log",
    "log_df",
    "disk_cache",
    "same_size",
    "same_nunique",
]
# Convert from: https://github.com/ejolly/engarde

from functools import wraps
import datetime as dt
import pandas as pd
import numpy as np
import deepdish as dd
from pathlib import Path


def log(func):
    """
    Log the type and shape/size/len of the output from a function

    Args:
        func (callable): any pure function (i.e, has no side-effects)
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        pass

    return wrapper


def log_df(func):
    """
    Log the shape and run time of a function that operates on a pandas dataframe

    Args:
        func (callable): a function that operates on a dataframe

    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        tic = dt.datetime.now()
        result = func(*args, **kwargs)
        time_taken = str(dt.datetime.now() - tic)
        print(f"Func {func.__name__} df shape={result.shape} took {time_taken}s")
        return result

    return wrapper


# TODO: Make cache dir in cwd and save cache there
# TODO: allow user to set cache location using kwargs
def disk_cache(threshold=60, autoload=True, index=False, verbose=False):
    """
    Save the result of a function to disk if it takes longer than threshold to run. Then on subsequent runs given the same arrangement of args and kwargs, first try to load the last result and return that, rather than rerunning the function, i.e. processing-time based memoization.

    Args:
        threshold (int, optional): threshold in seconds over which object is saved to disk. Defaults to 60.
        autoload (bool, optional): whether to try to load a previously persisted result if all args and kwargs match in a subsequent function call. Default to True;
        index (bool; optional): whether to incluce the index when saving a dataframe. Default to False
        verbose (bool; optional): print debug messages; Default to False
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            filtered_args = [
                e for e in args if not isinstance(e, (list, pd.DataFrame, np.ndarray))
            ]
            key = str((filtered_args, tuple(sorted(kwargs.items()))))
            key_csv = f"{func.__name__}_{key}.csv"
            key_h5 = f"{func.__name__}_{key}.h5"
            if autoload:
                if Path(key_csv).exists():
                    if verbose:
                        print("Returning cached result")
                    return pd.read_csv(key_csv)
                elif Path(key_h5).exists():
                    if verbose:
                        print("Returning cached result")
                    return dd.io.load(key_h5)
            if verbose:
                print("No cached result...executing")
            tic = dt.datetime.now()
            result = func(*args, **kwargs)
            time_taken = dt.datetime.now() - tic
            if time_taken.seconds > threshold:
                if isinstance(result, pd.DataFrame):
                    fname = f"{func.__name__}_{key}.csv"
                    result.to_csv(fname, index=index)
                else:
                    fname = f"{func.__name__}_{key}.h5"
                    dd.io.save(fname, result, compression="zlib")
                print(f"Exceeded threshold. Result cached to {fname}")
            return result

        return wrapper

    return decorator


# TODO: write me
def same_size(func, group_col):
    """
    Check if each group of group_col has the same dimensions after running a function on a dataframe

    Args:
        func (callable): a function that operates on a dataframe
        group_call (str): column name to group on in dataframe
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        pass

    return wrapper


# TODO: write me
def same_nunique(func, val_col, group_col):
    """
    Check if each group of group_col has the same number of unique values of val_col after running a function on a dataframe

    Args:
        func (callable): a function that operates on a dataframe
        val_col (str): column name to check for unique values in dataframe
        group_call (str): column name to group on in dataframe
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        pass

    return wrapper
