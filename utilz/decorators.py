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
from typing import Union, Any, Callable
import datetime as dt
import pandas as pd
import numpy as np
from pathlib import Path
from .io import load
from joblib import Memory


def _is_notebook() -> bool:
    """Helper function for show"""
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False


def show(func):
    """
    Print result of function call in addition to returning it

    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if result is None:
            # Return input
            to_show = args[0]
            to_return = to_show
        elif isinstance(result, pd.DataFrame):
            # print head, return result
            to_show = result.head()
            to_return = result
        else:
            # print and return result
            to_show = result
            to_return = to_show
        if _is_notebook():
            from IPython.display import display

            print_func = display
        else:
            print_func = print

        print_func(to_show)
        return to_return

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


def maybe(function):
    """
    A decorator that wraps a function which should take a kwarg called `out_file`. If
    `out_file` exists then it's loaded from disk, otherwise the wrapped function is
    called. If the wrapped function takes a kwarg `overwrite = True` then it always runs. You can also pass `loader_func = callable` to use a custom loading function

    Example:

    >>> @maybe
    >>> def mean_brain(subpath, **kwargs):
    >>> b = Brain_Data(subpath)
    >>> m = b.mean()
    >>> out_file = kwargs.get('out_file')
    >>> m.write(out_file)
    >>> return m

    >>> mean_brain(subpath, out_file='mean_brain.h5', loader_func=Brain_Data)

    """

    @wraps(function)
    def wrapper(*args, **kwargs):
        # get out_file from the wrapped function
        out_file = kwargs.get("out_file", None)

        # get out_file from the wrapped function
        overwrite = kwargs.get("overwrite", False)

        if out_file is None:
            raise ValueError(
                "out_file must be provided as a kwarg to the decorated function!"
            )

        out_file = Path(out_file)

        if out_file.exists() and not overwrite:
            print(f"Loading precomputed result from: {out_file}")
            load_kwargs = {
                k: v for k, v in kwargs.items() if k not in ["out_file", "overwrite"]
            }
            return load(out_file, **load_kwargs)

        return function(*args, **kwargs)

    return wrapper


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
