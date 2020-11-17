"""
Functions that exist purely to save boilerplate.
"""

__all__ = ["mpinit", "randdf"]

import numpy as np
import pandas as pd


def mpinit(figsize=(8, 6), subplots=(1, 1)):
    """
    Setup matplotlib subplots boilerplate

    Args:
        figsize (tuple, optional): Figure size. Defaults to (8, 6).
        subplots (tuple, optional): subplot grid size. Defaults to (1, 1).

    Returns:
        tuple ((Figure, Axes)): matplotlib figure handle and axes
    """
    if "plt" not in dir():
        import matplotlib.pyplot as plt
    f, ax = plt.subplots(*subplots, figsize=figsize)
    return f, ax


# TODO: test me
def randdf(size=(10, 3), column_names=["A", "B", "C"], func=None, *args, **kwargs):
    """
    Generate a dataframe with random data and alphabetic columns, Default to np.random.randn. Specify another function and size will be passed in as a kwarg.

    Args:
        size (tuple, optional): Defaults to (10,3).
        column_names (list, optional): Defaults to ["A","B","C"].
        func (callable, optional): function to generate data. Must take a kwarg "shape" that accepts a tuple; Default np.random.randn
        *args/**kwargs: arguments and keyword arguments to func
    """

    if len(column_names) != size[1]:
        raise ValueError("Length of column names must match number of columns")

    if func is None:
        data = np.random.rand(*size)
    else:
        data = func(*args, size=size, **kwargs)

    return pd.DataFrame(data, columns=column_names)
