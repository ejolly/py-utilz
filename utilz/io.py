"""
I/O Module for working with Paths
"""
__all__ = ["load", "save", "nbsave", "nbload"]
from pathlib import Path
from typing import Union, Any
import pandas as pd
import numpy as np
import deepdish as dd
import pickle
import json
from toolz.functoolz import memoize
import scrapbook as sb


# TODO: test me
def nbload(varname, fname, to_arr=False):
    """
    Load a variable previously saved in a notebook's json meta-data using scrapbook

    Args:
        varname (string): variable to load
        fname (string/Path): notebook path to load
        to_arr (bool, optional): whether to cast the loaded variable to a numpy array; Default False

    Returns:
        Any: the loaded variable
    """
    if isinstance(fname, Path):
        fname = str(Path)
    if to_arr:
        return np.array(sb.read_notebook(fname).scraps[varname].data)
    else:
        return sb.read_notebook(fname).scraps[varname].data


# TODO: Test me
# TODO: implement check to see if data is saved by calling nbload. Tricky thing is figuring out how to auto-get name of current notebook
def nbsave(var, varname):
    """
    Save a variable within a notebook's json meta-data using scrapbook. Auto-converts numpy arrays to lists before saving

    Args:
        var (Any): variable to save
        varname (string): what to call the variable in the notebook meta-data
    """
    if isinstance(var, np.ndarray):
        var = var.tolist()
    sb.glue(varname, var)


@memoize
def load(
    f: Union[Path, str],
    as_df: bool = False,
    as_arr: bool = False,
    as_str: bool = False,
    h5_key: str = "data",
    json_str: bool = False,
    pickle_encoding: str = "rb",
    verbose: bool = False,
    **kwargs,
) -> Any:
    """
    A handy dandy all-in-one loading function. Simply pass a Path object to a file (or a string) and you'll back a python object. Supported extensions are: .txt, .csv, .json, .p, .pickle, .h5, .hdf5, .gz

    Args:
        f (Path/str): name or path object to load
        as_df (bool, optional): treat a .hdf5, .h5 file as a Dataframe; Default False
        as_arr (bool, optional): treat a .txt file as a numpy array;
        Default False
        as_str (bool, optional): open txt as a single string instead of
        splitting on newlines; Default False
        json_str (bool, optional): treat a json file as a string (i.e use json.loads
        instead of json.load); Default False
        h5_key (str, optional): the key within the h5 file to load when if using as_df;
        Default 'data'
        pickle_encoding (str, optional): pickle encoding to use; Default 'rb'
        verbose (bool, optional): whether to print messages during load. Default False
        **kwargs: keyword arguments to pd.read_csv, np.loadtxt, dd.io.load, pickle, or open

    Returns:
        the loaded object
    """

    if isinstance(f, str):
        f = Path(f)
    if not isinstance(f, Path):
        raise TypeError("Input must be a string or Path object")

    supported_exts = [".txt", ".json", ".p", ".pickle", ".csv", ".h5", "hdf5", ".gz"]

    if f.suffix == ".csv":
        if verbose:
            print("csv file - using pandas")
        out = pd.read_csv(str(f), **kwargs)

    elif f.suffix == ".h5" or f.suffix == ".hdf5":
        if as_df:
            if verbose:
                print("h5 file as df - using pandas")
            out = pd.read_hdf(str(f), key=h5_key)
        else:
            if verbose:
                print("h5 file - using deepdish")
            out = dd.io.load(str(f))

    elif f.suffix == ".txt":
        if as_arr:
            if verbose:
                print("txt file - using numpy")
            out = np.loadtxt(str(f))
        else:
            if verbose:
                print("txt file using - open")
            with f.open() as file_handle:
                if as_str:
                    out = file_handle.read()
                else:
                    out = file_handle.readlines()

    elif f.suffix == ".p" or f.suffix == ".pickle":
        if verbose:
            print("pickle file - using pickle")
        out = pickle.load(open(str(f), pickle_encoding))

    elif f.suffix == ".json":
        if verbose:
            print("json file - using pickle")
        with f.open() as file_handle:
            if json_str:
                out = json.loads(file_handle.read())
            else:
                out = json.load(file_handle)

    elif f.suffix == ".gz":
        if verbose:
            print("gz file - using numpy")
        out = np.loadtxt(str(f))

    else:
        raise TypeError(f"File must end in one of: {supported_exts}")

    return out


# TODO: Write me
def save(f: Union[Path, str]) -> None:
    """
    A handy dandy all-in-one saving function. Simply pass a Path object to a file (or a string) and it will be saved based upon the file *extension* you provide. Suported extensions are : .txt, .csv, .json, .p, .pickle, .h5, .hd5f, .gz

    Args:
        f (Path/str): complete filepath to save to including extension
    """
    pass
