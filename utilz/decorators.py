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
    "log",
    "timeit",
    "maybe",
    "expensive",
]

from functools import wraps
from typing import Union, Any
import datetime as dt
import pandas as pd
import numpy as np
from pathlib import Path
from .io import load
from joblib import Memory


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


def timeit(func):
    """
    Log the run time of a function

    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        tic = dt.datetime.now()
        result = func(*args, **kwargs)
        time_taken = str(dt.datetime.now() - tic)
        print(f"{func.__name__}, took {time_taken}s")
        return result

    return wrapper


def maybe(
    fpath: Union[str, Path],
    force: bool = False,
    as_arr: bool = False,
    as_str: bool = False,
    verbose: bool = False,
) -> Any:
    """
    Run the decorated `func` only if `fpath` doesn't exist or if it isn't an empty
    directory. If `fpath` exists will load the file from disk or if `fpath` is a directory,
    will return the results of globbing the directory for all files

    Args:
        fpath (Path/str): filename or dirname to check existence for
        force (bool, optional): always run the function even if filepath exists,
        possibly overwriting filepath (based on whatever func does internally). Defaults
        to False.
        as_arr (bool, optional): treat a .txt file as a numpy array;
        Default False
        as_str (bool, optional): open txt/json as a single string instead of
        splitting on newlines; Default False
        verbose (bool, optional): whether to print messages during load. Default False
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not force and fpath.exists():
                if fpath.is_file() or (fpath.is_dir() and any(fpath.iterdir())):
                    print("Exists: loading previously saved output")
                    return load(
                        fpath,
                        as_str=as_str,
                        as_arr=as_arr,
                        verbose=verbose,
                    )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def expensive(
    force: bool = False,
) -> Any:
    """
    A decorator that wraps `joblib.Memory` for caching the results of a function to disk.
    This is useful for expensive functions that take a while to compute, as rerunning
    them will simply load the last results from disk.

    Args:
        force (bool, optional): clear the cache before rerunning; Default False
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            memory = Memory("./cachedir")
            if force:
                memory.clear()
            func_cached = memory.cache(func)
            return func_cached(*args, **kwargs)

        return wrapper

    return decorator
