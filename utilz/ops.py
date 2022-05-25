"""
Functional tools

---
"""
__all__ = ["check_random_state", "mapcat"]

from joblib import delayed, Parallel
from ._utils import ProgressParallel
import numpy as np
import pandas as pd
from typing import Union, Any
from collections.abc import Callable, Iterable
from itertools import chain
from inspect import signature
from tqdm import tqdm
from typeguard import typechecked

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


# Helper used by mapcat
def _pmap(
    func: Callable,
    iterme: Iterable,
    enum: bool = False,
    seeds: Union[None, list] = None,
    n_jobs: int = 2,
    backend: Union[None, str] = None,
    progressbar: bool = True,
    verbose: int = 0,
) -> Any:

    # Initialize joblib parallelizer
    if progressbar:
        parfor = ProgressParallel(
            prefer=backend, n_jobs=n_jobs, verbose=verbose, total=len(iterme)
        )
    else:
        parfor = Parallel(prefer=backend, n_jobs=n_jobs, verbose=verbose)

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
    return out


@typechecked
def mapcat(
    func: Union[Callable, None],
    iterme: Iterable,
    concat: bool = True,
    enum: bool = False,
    random_state: Union[bool, int, np.random.RandomState] = False,
    n_jobs: int = 1,
    backend: Union[None, str] = None,
    axis: Union[None, int] = None,
    ignore_index: bool = True,
    pbar: bool = True,
    verbose: int = 0,
):
    """
    Super-power your `for` loops with a progress-bar and optional *reproducible*
    parallelization!

    **map**s `func` to `iterme` and con**cat**enates the result into a single, list,
    DataFrame or array. Includes a progress-bar powered by `tqdm`.

    Supports parallelization with `jobllib.Parallel` multi-processing by setting `n_jobs > 1`. Progress-bar *accurately* tracks parallel jobs!

    `iterme` can be a list of elements, list of DataFrames, list of arrays, or list of lists. List of lists up to 2 deep will be flattened to single
    list.

    See the examples below for interesting use cases beyond standard looping!

    Args:
        func (callable): function to map
        iterme (iterable): an iterable for which each element will be passed to func
        concat (bool): if `func` returns an interable, will try to flatten iterables
        into a single list, array, or dataframe based on `axis`; Default True
        enum (bool, optional): whether the value of the current iteration should be passed to `func` as the special kwarg `idx`. Make sure `func` can handle a kwarg named `idx`. Default False
        random_state (bool/int, optional): whether a randomly initialized seed should be
        passed to `func` as the special kwarg `random_state`. The function should pass
        this seed to the `utilz.check_random_state` helper to generate a random number
        generator for all computations rather than relying on `np.random`
        n_jobs (int, optional): number of cpus/threads; Default 1 (no parallel)
        backend (str, optional): Only applies if `n_jobs > 1`. See `joblib.Parallel` for
        options; Default None which uses `loky`
        axis (int; optional): what axis to concatenate over; Default 0 (first)
        ignore_index (bool; optional): ignore the index when combining DataFrames;
        Default True
        pbar (bool, optional): whether to use tqdm to show a progressbar; Default
        True
        verbose (int): `joblib.Parallel` verbosity. Default 0

    Examples:
        >>> # Just like map
        >>>  out = mapcat(lambda x: x * 2, [1, 2, 3, 4])

        >>> # Concatenating nested lists
        >>> data = [[1, 2], [3, 4]]
        >>> out = mapcat(None, data)

        >>> # Load multiple files into a single dataframe
        >>> out = mapcat(pd.read_csv, ["file1.txt", "file2.txt", "file3.txt"])

        >>> # Parallelization with randomness
        >>> def f_random(x, random_state=None):
        >>>     random_state = check_random_state(random_state)
        >>>     sleep(0.5)
        >>>     # Use the random state's number generator rather than np.random
        >>>     return x + random_state.rand()
        >>>
        >>> # Now set a random_state in mapcat to reproduce the parallel runs
        >>> # It doesn't pass the value, but rather generates a reproducible list
        >>> # of seeds that are passed to each function execution
        >>> out = mapcat(f_random, [1, 1, 1, 1, 1], n_jobs=2, random_state=1)

    """

    if func is None:
        # No-op if no function
        op = iterme
    else:
        func_args = list(signature(func).parameters.keys())

        if enum and "idx" not in func_args:
            raise ValueError(
                "Function must accept a keyword argument named 'idx' that accepts an integer if enum is True"
            )

        if random_state is not False:
            if "random_state" not in func_args:
                raise ValueError(
                    "Function must have a keyword argument called 'random_state' if random_state is not False"
                )

            random_state = None if random_state is True else random_state

            # Generate a list of random ints, that themselves are seeded by random_state
            # and passed to func
            seeds = check_random_state(random_state).randint(MAX_INT, size=len(iterme))
        else:
            seeds = None

        if n_jobs > 1 or n_jobs < 1:
            # Parallel loop
            op = _pmap(
                func,
                iterme,
                enum,
                seeds,
                n_jobs,
                backend,
                pbar,
                verbose,
            )
        else:
            # Normal loop because its faster than joblib.Parallel with n_jobs == 1 (no
            # overhead cost)
            iterator = tqdm(iterme) if pbar else iterme

            if seeds is None and not enum:
                op = [func(elem) for elem in iterator]
            elif seeds is None and enum:
                op = [func(elem, idx=i) for i, elem in enumerate(iterator)]
            elif seeds is not None and not enum:
                op = [
                    func(elem, random_state=seed) for elem, seed in zip(iterator, seeds)
                ]
            elif seeds is not None and enum:
                op = [
                    func(elem, random_state=seed, idx=i)
                    for i, (elem, seed) in enumerate(zip(iterator, seeds))
                ]

    if concat:
        return _concat(op, iterme, axis, ignore_index)
    return op


# Helper used by mapcat
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
