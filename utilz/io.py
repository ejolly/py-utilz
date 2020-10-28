"""
I/O Module for working with Paths
"""

from pathlib import Path
from typing import Union, Any
import pandas as pd
import numpy as np
import deepdish as dd
import pickle
import json


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
    A handy dandy all-in-one loading function. Simple pass a Path object to a file and
    you'll back a python object. Supported extensions are: .txt, .csv, .json, .p,
    .pickle, .h5, .hdf5, .gz

    Args:
        f (str/Path): name or path object to load
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
        out (Any): the loaded object
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
