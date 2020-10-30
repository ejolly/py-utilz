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
from typing import Any
from hashlib import sha256
from json import dumps
from inspect import getcallargs


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


def _hashobj(obj):
    if isinstance(obj, pd.DataFrame):
        return sha256(obj.to_json().encode()).hexdigest()
    elif isinstance(obj, np.ndarray):
        return sha256(obj.tostring()).hexdigest()
    elif isinstance(obj, (list, tuple)):
        return sha256(str(obj).encode()).hexdigest()
    else:
        return obj


def disk_cache(
    threshold: int = 30,
    autoload: bool = True,
    index: bool = False,
    save_dir: str = ".utilz_cache",
) -> Any:
    """
    Save the result of a function to disk if it takes longer than threshold to run. Then on subsequent runs given the same arrangement of args and kwargs, first try to load the last result and return that, rather than rerunning the function, i.e. processing-time based memoization.

    Args:
        threshold (int, optional): threshold in seconds over which object is saved to disk. Defaults to 30.
        autoload (bool, optional): whether to try to load a previously persisted result if all args and kwargs match in a subsequent function call. Default to True;
        index (bool; optional): whether to incluce the index when saving a dataframe. Default to False
        save_dir (str; optional): location to cache results; Default '.utilz_cache'
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            saved_inputs = dict(sorted(getcallargs(func, *args, **kwargs).items()))
            cache_dir = Path(save_dir)
            if not cache_dir.exists():
                cache_dir.mkdir()
            inputs = {}
            for k, v in saved_inputs.items():
                if k == "args":
                    new_v = [_hashobj(e) for e in v]
                    inputs[k] = new_v
                elif k == "kwargs":
                    new_v = {kk: _hashobj(vv) for kk, vv in v.items()}
                    inputs[k] = new_v
                else:
                    inputs[k] = _hashobj(v)
            key = (
                dumps(inputs)
                .replace('"', "")
                .replace("{", "")
                .replace("}", "")
                .replace("[", "")
                .replace("]", "")
                .replace(":", "__")
                .replace(" ", "")
                .replace(",", "--")
            )
            key_csv = f"{func.__name__}___{key}.csv"
            key_csv = cache_dir.joinpath(key_csv)
            key_h5 = f"{func.__name__}___{key}.h5"
            key_h5 = cache_dir.joinpath(key_h5)
            if autoload:
                if Path(key_csv).exists():
                    print("Returning previously saved result")
                    return pd.read_csv(key_csv)
                elif Path(key_h5).exists():
                    print("Returning previously saved result")
                    return dd.io.load(key_h5)
            tic = dt.datetime.now()
            result = func(*args, **kwargs)
            time_taken = dt.datetime.now() - tic
            if time_taken.seconds > threshold:
                Path("__utilz_cache__").mkdir()
                if isinstance(result, pd.DataFrame):
                    result.to_csv(str(key_csv), index=index)
                    print(f"Exceeded compute time. Result saved to {key_csv}")
                else:
                    dd.io.save(str(key_h5), result, compression="zlib")
                    print(f"Exceeded compute time. Result saved to {key_h5}")
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
