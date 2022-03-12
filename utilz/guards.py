"""
Custom guards for defensive data analysis compatible with [bulwark](https://bulwark.readthedocs.io/en/latest/index.html).

Intended usage is as Python decorators:

```python
from utilz.guards import log_df

@log_df
def myfunc(df):
    do some stuff...
```

---
"""
__all__ = [
    "show",
    "copy",
    "log",
    "time",
    "maybe",
    "disk_cache",
    "same_shape",
    "same_nunique",
]
# Convert from: https://github.com/ejolly/engarde

from functools import wraps
from typing import Union, Any
import datetime as dt
import pandas as pd
import numpy as np
import deepdish as dd
from pathlib import Path
from copy import deepcopy
from hashlib import sha256
from json import dumps
from inspect import getcallargs
from .io import load


def show(func):
    """
    Print result of function call in addition to returning it

    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print(result)
        return result

    return wrapper


def copy(func):
    """
    Make a copy of the first argument before passing it to func to avoid modifying the argument itself, e.g. a list, dataframe, array, etc. Does not affect other arguments

    """

    @wraps(func)
    def wrapper(arg1, *args, **kwargs):
        if hasattr(arg1, "copy"):
            copied = arg1.copy()
        else:
            copied = deepcopy(arg1)
        result = func(copied, *args, **kwargs)
        return result

    return wrapper


def log(func):
    """
    Log the type and shape/size/len of the output from a function

    """

    @wraps(func)
    def wrapper(arg1, *args, **kwargs):
        if isinstance(arg1, pd.DataFrame):
            print(f"before {func.__name__}, {arg1.shape}, df")
        elif isinstance(arg1, np.ndarray):
            print(f"before {func.__name__}, {arg1.shape}, np")
        elif isinstance(arg1, list):
            print(f"before {func.__name__}, {len(arg1)}, []")
        elif isinstance(arg1, dict):
            print(f"bebfore {func.__name__}, {len(arg1.keys())}, {{}}")
        result = func(arg1, *args, **kwargs)
        if isinstance(result, pd.DataFrame):
            print(f"after {func.__name__}, {result.shape}, df")
        elif isinstance(result, np.ndarray):
            print(f"after {func.__name__}, {result.shape}, np")
        elif isinstance(result, list):
            print(f"after {func.__name__}, {len(result)}, []")
        elif isinstance(result, dict):
            print(f"after {func.__name__}, {len(result.keys())}, {{}}")
        return result

    return wrapper


def time(func):
    """
    Log the run time of a function

    Args:
        func (callable): any pure function (i.e, has no side-effects)

    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        tic = dt.datetime.now()
        result = func(*args, **kwargs)
        time_taken = str(dt.datetime.now() - tic)
        print(f"{func.__name__}, took {time_taken}s")
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


def maybe(fpath: Union[str, Path], force: bool = False) -> Any:
    """
    Run the decorated `func` only if fpath doesn't exist. If it does exist calls
    io.load(fpath), unless force == True

    Args:
        fpath (Path/str): filename or path to check existence for
        force (bool, optional): always run the function even if filepath exists, possibly overwriting filepath (based on whatever func does internally). Defaults to False.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not force and fpath.exists():
                print("loading previously saved file")
                return load(fpath)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def disk_cache(
    threshold: int = 30,
    autoload: bool = True,
    index: bool = False,
    save_dir: str = ".utilz_cache",
) -> Any:
    """
    Save the result of a function to disk if it takes longer than `threshold` to run. Then on subsequent runs given the same `args` and `kwargs`, first try to load the last result and return that, rather than rerunning the function, i.e. processing-time based memoization. The resulting file is saved to `.utilz_cache/funcname___arg1__arg1val--arg2__arg2val__kwarg1__kwarg1val--kwarg2__kwarg2val.{csv/h5}`.

    Very similar in spirit to `@memory.cache` decorator in `joblib` but instead uses csv's to persist Dataframes and hdf5 to persist everything else rather than pickles. Also works better in combination with the `@curry` decorator from `toolz/cytoolz`

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
                    print(f"Returning {func.__name__} cached result")
                    return pd.read_csv(key_csv)
                elif Path(key_h5).exists():
                    print(f"Returning {func.__name__} cached result")
                    return dd.io.load(key_h5)
            tic = dt.datetime.now()
            result = func(*args, **kwargs)
            time_taken = dt.datetime.now() - tic
            if time_taken.seconds > threshold:
                if isinstance(result, pd.DataFrame):
                    result.to_csv(str(key_csv), index=index)
                    print(f"Exceeded compute time. Result cached to {key_csv}")
                else:
                    dd.io.save(str(key_h5), result, compression="zlib")
                    print(f"Exceeded compute time. Result cached to {key_h5}")
            return result

        return wrapper

    return decorator


# TODO: write me
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
def same_nunique(func: callable, val_col: str, group_col: str):
    """
    Check if each group of `group_col` has the same number of unique values of `val_col` after running a function on a dataframe

    Args:
        func (callable): a function that operates on a dataframe
        val_col (str): column name to check for unique values in dataframe
        group_col (str): column name to group on in dataframe
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        pass

    return wrapper
