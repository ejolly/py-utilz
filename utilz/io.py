"""
I/O Module
"""
__all__ = ["load"]
from pathlib import Path
from typing import Union, Any
import pandas as pd
import numpy as np
import pickle
import json
from toolz.functoolz import memoize


@memoize
def load(
    f: Union[Path, str],
    as_arr: bool = False,
    as_str: bool = False,
    verbose: bool = False,
    **kwargs,
) -> Any:
    """
    A handy dandy all-in-one loading function. Simply pass a Path object to a file (or a
    string) and you'll back a python object based on the file-extension:

    - `.csv`: `pd.Dataframe`
    - `.txt`: `np.ndarray`, `list[str]` (lines a file), or `str` (all file contents)
    - `.p/.pickle`: output of `pickle.load`
    - `.json`: `str` or `dict`

    Args:
        f (Path/str): name or path object to load
        as_arr (bool, optional): treat a .txt file as a numpy array;
        Default False
        as_str (bool, optional): open txt/json as a single string instead of
        splitting on newlines; Default False
        verbose (bool, optional): whether to print messages during load. Default False
        **kwargs: keyword arguments to `pd.read_csv` or `np.loadtxt`

    Returns:
        the loaded object
    """

    if isinstance(f, str):
        f = Path(f)
    if not isinstance(f, Path):
        raise TypeError("Input must be a string or Path object")

    if f.is_dir():
        return list(f.glob("*"))

    # TODO: add support for np.save .npz files
    supported_exts = [".txt", ".json", ".p", ".pickle", ".csv"]

    if f.suffix == ".csv":
        if verbose:
            print("csv file - using pandas")
        return pd.read_csv(str(f), **kwargs)

    if f.suffix == ".txt":
        if as_arr:
            if verbose:
                print("txt file - using numpy")
            return np.loadtxt(str(f), **kwargs)
        else:
            if verbose:
                print("txt file using - open")
            with f.open() as file_handle:
                if as_str:
                    return file_handle.read()
                else:
                    return file_handle.readlines()

    if f.suffix == ".p" or f.suffix == ".pickle":
        if verbose:
            print("pickle file - using pickle")
        with f.open(mode="rb") as file_handle:
            return pickle.load(file_handle)

    if f.suffix == ".json":
        if verbose:
            print("json file - using pickle")
        with f.open() as file_handle:
            if as_str:
                return json.loads(file_handle.read())
            else:
                return json.load(file_handle)

    raise TypeError(f"File must end in one of: {supported_exts}")
