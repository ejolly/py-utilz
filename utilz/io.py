"""
I/O Module
"""
__all__ = ["load", "crawl"]
from pathlib import Path
from typing import Union, Any, List
import pandas as pd
import numpy as np
import pickle
import json
from toolz.functoolz import memoize
from warnings import warn
from .ops import mapcat, filtercat, sort
from toolz import pipe
from fnmatch import fnmatchcase


@memoize
def load(
    f: Union[Path, str],
    as_arr: bool = False,
    as_str: bool = False,
    verbose: bool = False,
    assert_notempty: bool = True,
    **kwargs,
) -> Any:
    """
    A handy dandy all-in-one loading function. Simply pass a Path object to a file (or a
    string) and you'll back a python object based on the file-extension:

    - `.csv`: `pd.Dataframe`
    - `.p/.pickle`: output of `pickle.load`
    - `.json`: `str` or `dict`
    - `.txt`: `np.ndarray`, `list[str]` (lines a file), or `str` (all file contents)
    - other file-extensions are attempted to be loaded like `.txt` files

    Args:
        f (Path/str): name or path object to load
        as_arr (bool, optional): treat a .txt file as a numpy array;
        Default False
        as_str (bool, optional): open txt/json as a single string instead of
        splitting on newlines; Default False
        assert_notempty(bool, optional): make sure the output is not Falsey (e.g. empty
        array, dataframe, string, list); Default True
        verbose (bool, optional): whether to print messages during load. Default False
        **kwargs: keyword arguments to `pd.read_csv` or `np.loadtxt`
        assert_notempty (bool, optional): raise an error if the returned output is
        empty; Default True

    Returns:
        the loaded object
    """

    if isinstance(f, str):
        f = Path(f)
    if not isinstance(f, Path):
        raise TypeError("Input must be a string or Path object")

    if f.is_dir():
        out = list(f.glob("*"))

    # TODO: add support for np.save .npz files

    elif f.suffix == ".csv":
        if verbose:
            print("csv file - using pandas")
        out = pd.read_csv(str(f), **kwargs)

    elif f.suffix == ".p" or f.suffix == ".pickle":
        if verbose:
            print("pickle file - using pickle")
        with f.open(mode="rb") as file_handle:
            out = pickle.load(file_handle)

    elif f.suffix == ".json":
        if verbose:
            print("json file - using pickle")
        with f.open() as file_handle:
            if as_str:
                out = json.loads(file_handle.read())
            else:
                out = json.load(file_handle)

    else:
        if verbose and f.suffix != ".txt":
            warn(f"{f.suffix} not supported treating as .txt file...")

        if as_arr:
            if verbose:
                print("txt file - using numpy")
            out = np.loadtxt(str(f), **kwargs)
        else:
            if verbose:
                print("txt file using - open")
            with f.open() as file_handle:
                if as_str:
                    out = file_handle.read()
                else:
                    out = file_handle.readlines()

    if assert_notempty and len(out) == 0:
        raise AssertionError("Loaded data is empty!")

    return out


def crawl(
    where: Union[str, Path] = ".",
    ignore_git: bool = True,
    respect_gitignore: bool = True,
    ignore: Union[None, list, str] = None,
) -> List:
    """
    Crawls a folder and returns a list of Path objects containing folders and files
    while respecting gitignore files if present an any additional ignore file names or patterns

    Args:
        where (Union[str, Path], optional): location to glob in. Defaults to ".".
        ignore_git (bool, optional): ignore `.git` folder. Defaults to True.
        respect_gitignore (bool, optional): read and ignore all files and patterns in
        `.gitignore` file. Defaults to True.
        ignore (Union[None, list, str], optional): additional files or patterns to ignore. Defaults to None.

    Returns:
        List: _description_
    """

    if ignore is None:
        ignore = []
    elif isinstance(ignore, str):
        ignore = [ignore]
    ignore_list = [".git"] if ignore_git else []
    if respect_gitignore:
        ignore_list += mapcat(lambda s: s.strip("\n"), load(".gitignore"))

    ignore_list += ignore
    # Split glob patterns and regular file/folder name matches
    globs, nonglobs = filtercat("*", ignore_list, invert="split")

    # Generator for all files
    out = Path(where)
    files = out.rglob("*")

    # Filter out folders and file names from gitignore
    # Filter out glob patters from gitignore
    return pipe(
        files,
        filtercat(nonglobs, invert=True),
        filtercat(lambda f: any(fnmatchcase(str(f), g) for g in globs), invert=True),
        sort,
    )
