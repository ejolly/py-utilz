"""
Functions that exist purely to save boilerplate.
"""

__all__ = ["randdf"]

import pandas as pd
from typing import Union
import string
from itertools import cycle
from .ops import check_random_state


def randdf(
    size: tuple = (10, 3),
    columns: Union[list, None] = None,
    func=None,
    random_state=None,
    *args: any,
    **kwargs: any,
):
    """
    Generate a dataframe with random data and alphabetic columns, Default to np.random.randn. Specify another function and size will be passed in as a kwarg.

    Args:
        size (tuple, optional): Defaults to (10,3).
        columns (list, optional): Defaults numbered capital letters like an excel spreadsheet
        func (callable, optional): function to generate data. Must take a kwarg "size"
        that accepts a tuple; Default np.random.randn
        random_state (None, int, np.RandomState): if None, return the RandomState
        singleton used by np.random. If int, return a new RandomState instance seeded
        with seed. If seed is already a RandomState instance, return it. Otherwise raise
        ValueError.
        *args: positional arguments to func
        **kwargs: keyword arguments to func
    """

    if columns is not None and len(columns) != size[1]:
        raise ValueError("Length of column names must match number of columns")

    rng = check_random_state(random_state)
    if func is None:
        data = rng.rand(*size)
    else:
        data = func(*args, size=size, **kwargs)

    if columns is None:
        letters = cycle(string.ascii_uppercase)
        counter = 1
        columns = []
        for i in range(size[1]):
            if i > 0 and i % 26 == 0:
                counter += 1
            columns.append(f"{next(letters)}{counter}")

    return pd.DataFrame(data, columns=columns)
