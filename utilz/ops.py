"""
Functional tools

---
"""
__all__ = ["check_random_state", "pmap", "mapcat"]

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from typing import Union, Any
from collections.abc import Callable, Iterable
from itertools import chain
from inspect import signature

MAX_INT = np.iinfo(np.int32).max


def check_random_state(seed=None):
    """Turn seed into a np.random.RandomState instance. Note: credit for this code goes entirely to `sklearn.utils.check_random_state`. Using the source here simply avoids an unecessary dependency.

    Args:
        seed (None, int, np.RandomState): if seed is None, return the RandomState singleton used by np.random. If seed is an int, return a new RandomState instance seeded with seed. If seed is already a RandomState instance, return it. Otherwise raise ValueError.
    """

    import numbers

    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState" " instance" % seed
    )


def pmap(
    func: Callable,
    iterme: Iterable,
    enum: bool = False,
    concat: bool = True,
    axis: Union[None, int] = None,
    ignore_index: bool = True,
    random_state: Union[bool, int, np.random.RandomState] = False,
    n_jobs: int = 2,
    backend: Union[None, str] = None,
    verbose: int = 0,
) -> Any:
    """
    Map a `func` to `iterme` using parallelization via joblib. Note the only difference between `pmap` and `prep` is that that `pmap` explicitly operates on an iterable, such that the input to `func` changes each time (each element of `iterme`); where as `prep` just repeatedely executes `func` for `n_iter` operations with optional args/kwargs that are the same for each run of `func`.

    Args:
        func (callable): function to run
        iterme (iterable): an iterable for which each element will be passed to func
        n_jobs (int, optional): number of cpus/threads; Default -1 (all cpus/threads)
        enum (bool, optional): whether the value of the current iteration should be passed to func as the special kwarg 'idx'. Make sure func can handle a kwarg named 'idx'. Default False
        random_state (bool/int, optional): whether a randomly initialized seed should be
        passed to func as the special kwarg 'random_state'. The function should pass
        thiseed to the `utilz.check_random_state` helper to generate a random number
        generator for all computations rather than relying on np.random
        backend (str, optional): see `joblib.Parallel` for options; Default None which
        uses 'loky'
        verbose (int): joblib.Parallel verbosity. Default 0

    """

    if n_jobs == 1:
        return mapcat(func, iterme, concat, axis, ignore_index)
    # Initialize joblib parallelizer
    parfor = Parallel(prefer=backend, n_jobs=n_jobs, verbose=verbose)

    func_args = list(signature(func).parameters.keys())
    if random_state is not False:
        if "random_state" not in func_args:
            raise ValueError(
                "Function must have a keyword argument called 'random_state' if random_state is not False"
            )

        random_state = None if random_state is True else random_state

        # Generate a list of random ints, that themselves are seeded by random_state and
        # will be passed to func
        seeds = check_random_state(random_state).randint(MAX_INT, size=len(iterme))
    else:
        seeds = None

    if enum and "idx" not in func_args:
        raise ValueError(
            "Function must accept a keyword argument named 'idx' that accepts an integer if enum is True"
        )

    if seeds is None and not enum:
        call_list = [delayed(func)(elem) for elem in iterme]
    elif seeds is None and enum:
        call_list = [delayed(func)(elem, idx=i) for i, elem in enumerate(iterme)]
    elif seeds is not None and not enum:
        call_list = [
            delayed(func)(elem, random_state=seed) for elem, seed in zip(iterme, seeds)
        ]
    elif seeds is not None and enum:
        call_list = [
            delayed(func)(elem, random_state=seed, idx=i)
            for i, (elem, seed) in enumerate(zip(iterme, seeds))
        ]

    out = parfor(call_list)
    if concat:
        return _concat(out, iterme, axis, ignore_index)
    return out


def mapcat(func, iterme, concat=True, axis=None, ignore_index=True):
    """
    **map**s `func` to `iterme` and con**cat**enates the result into a single, list, DataFrame or array. `iterme` can be a list of elements, list of DataFrames, list of arrays, or list of lists. List of lists up to 2 deep will be flattened to single list.

    A a few interesting use cases include:

    - Passing None for the value of func acts as a shortcut to flatten nested lists.
    - Using in place of `map` acts like a call to `list(map(...))`
    - Passing in `pd.read_csv` to list of files to get back a single dataframe

    Args:
        func (callable): function to apply. If None, will attempt to flatten a nested list
        iterme (iterable): an iterable for which func will be called on each element
        concat (bool): if func returns an interable, will try to flatten iterables into
        a single list, array, or dataframe based on axis; Default True
        axis (int; optional): what axis to concatenate over; Default 0 (first)
        ignore_index (bool; optional): ignore the index when combining DataFrames; Default True
    """

    if func is not None:
        op = list(map(func, iterme))
    else:
        # No-op if no function
        op = iterme

    if concat:
        return _concat(op, iterme, axis, ignore_index)
    return op


# Helper concat function used by mapcat and pmap
def _concat(op, iterme, axis, ignore_index):

    try:
        if isinstance(op[0], pd.DataFrame):
            return pd.concat(
                op, axis=0 if axis is None else axis, ignore_index=ignore_index
            )

        if isinstance(op[0], np.ndarray) or isinstance(iterme, np.ndarray):
            try:
                if axis is None:
                    return np.array(op)
                return np.concatenate(op, axis=axis)
            except np.AxisError as _:
                return np.stack(op, axis=axis - 1)
            except ValueError as _:
                return np.array(op)
        try:
            return list(chain.from_iterable(op))
        except TypeError as _:
            return op

    except Exception as e:
        print(e)
        return op
