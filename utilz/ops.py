"""
Common data operations and transformations often on pandas dataframes

---
"""
__all__ = ["random_seed", "norm_by_group", "splitdf", "pmap", "prep", "mapcat"]

# from cytoolz import curry
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import Union, Any
from collections.abc import Callable, Iterable
from warnings import warn
from itertools import chain

MAX_INT = np.iinfo(np.int32).max


def random_seed(seed):
    """Turn seed into a np.random.RandomState instance. Note: credit for this code goes entirely to `sklearn.utils.check_random_state`. Using the source here simply avoids an unecessary dependency.

    Args:
        seed (None, int, np.RandomState): iff seed is None, return the RandomState singleton used by np.random. If seed is an int, return a new RandomState instance seeded with seed. If seed is already a RandomState instance, return it. Otherwise raise ValueError.
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


# @curry
def norm_by_group(df, grpcols, valcol, center=True, scale=True):
    """
    Normalize values in a column separately per group

    Args:
        df (pd.DataFrame): input dataframe
        grpcols (str/list): grouping col(s)
        valcol (str): value col
        center (bool, optional): mean center. Defaults to True.
        scale (bool, optional): divide by standard deviation. Defaults to True.
    """

    def _norm(dat, center, scale):
        if center:
            dat = dat - dat.mean()
        if scale:
            dat = dat / dat.std()
        return dat

    return df.groupby(grpcols)[valcol].transform(_norm, center, scale)


def pmap(
    func: Callable,
    iterme: Iterable,
    func_args: list = None,
    n_jobs: int = -1,
    loop_idx: bool = True,
    loop_random_seed: bool = False,
    backend: str = "processes",
    verbose: int = 0,
    seed: Union[None, int, np.random.RandomState] = None,
) -> Any:
    """
    Map a `func` to `iterme` using parallelization via joblib. Note the only difference between `pmap` and `prep` is that that `pmap` explicitly operates on an iterable, such that the input to `func` changes each time (each element of `iterme`); where as `prep` just repeatedely executes `func` for `n_iter` operations with optional args/kwargs that are the same for each run of `func`.

    Args:
        func (callable): function to run
        iterme (iterable): an iterable for which each element will be passed to func
        func_args (list/dict/None): additional arguments to the function provided as a list for unnamed args or a dict for named kwargs. If None, assumes func takes no arguments excepted loop_idx_available (if its True); Default None
        n_jobs (int, optional): number of cpus/threads; Default -1 (all cpus/threads)
        loop_idx (bool, optional): whether the value of the current iteration should be passed to func as the special kwarg 'idx'. Make sure func can handle a kwarg named 'idx'. Default True
        loop_random_seed (bool, optional): whether a randomly initialized seed should be passed to func as the special kwarg 'seed'. If func depends on any randomization (e.g. np.random) this should be set to True to ensure that parallel processes/threads use independent random seeds. Make sure func can handle a kwarg named 'seed' and utilize it for randomization. See example. Default False.
        backend (str, optional): 'processes' or 'threads'. Use 'threads' when you know you function releases Python's Global Interpreter Lock (GIL); Default 'cpus'
        verbose (int): joblib.Parallel verbosity. Default 0
        seed (int/None): random seed for reproducibility

    """

    if backend not in ["processes", "threads"]:
        raise ValueError("backend must be one of cpu's threads")

    parfor = Parallel(prefer=backend, n_jobs=n_jobs, verbose=verbose)
    if loop_random_seed:
        seeds = random_seed(seed).randint(MAX_INT, size=len(iterme))

    if func_args is None:
        if loop_idx:
            if loop_random_seed:
                out = parfor(
                    delayed(func)(e, **{"idx": i, "seed": seeds[i]})
                    for i, e in enumerate(iterme)
                )
            else:
                out = parfor(
                    delayed(func)(e, **{"idx": i}) for i, e in enumerate(iterme)
                )
        else:
            if loop_random_seed:
                out = parfor(
                    delayed(func)(e, **{"seed": seeds[i]}) for i, e in enumerate(iterme)
                )
            else:
                out = parfor(delayed(func)(e for e in iterme))
    else:
        if loop_idx:
            if loop_random_seed:
                if isinstance(func_args, list):
                    out = parfor(
                        delayed(func)(e, *func_args, **{"idx": i, "seed": seeds[i]})
                        for i, e in enumerate(iterme)
                    )
                elif isinstance(func_args, dict):
                    out = parfor(
                        delayed(func)(e, **func_args, **{"idx": i, "seed": seeds[i]})
                        for i, e in enumerate(iterme)
                    )
                else:
                    raise TypeError("func_args must be a list or dict")
            else:
                if isinstance(func_args, list):
                    out = parfor(
                        delayed(func)(e, *func_args, **{"idx": i})
                        for i, e in enumerate(iterme)
                    )
                elif isinstance(func_args, dict):
                    out = parfor(
                        delayed(func)(e, **func_args, **{"idx": i})
                        for i, e in enumerate(iterme)
                    )
                else:
                    raise TypeError("func_args must be a list or dict")
        else:
            if loop_random_seed:
                if isinstance(func_args, list):
                    out = parfor(
                        delayed(func)(e, *func_args, **{"seed": seeds[i]})
                        for i, e in enumerate(iterme)
                    )
                elif isinstance(func_args, dict):
                    out = parfor(
                        delayed(func)(e, **func_args, **{"seed": seeds[i]})
                        for i, e in enumerate(iterme)
                    )
                else:
                    raise TypeError("func_args must be a list or dict")
            else:
                if isinstance(func_args, list):
                    out = parfor(delayed(func)(e, *func_args) for e in iterme)
                elif isinstance(func_args, dict):
                    out = parfor(delayed(func)(e, **func_args) for e in iterme)
                else:
                    raise TypeError("func_args must be a list or dict")
    return out


def prep(
    func,
    func_args=None,
    n_iter=100,
    n_jobs=-1,
    loop_idx=True,
    loop_random_seed=False,
    backend="processes",
    progress=True,
    verbose=0,
    seed=None,
):
    """
    Call a `func` for `n_iter` using parallelization via `joblib`. Note the only difference between `pmap` and `prep` is that that `pmap` explicitly operates on an iterable, such that the input to `func` changes each time (each element of `iterme`); where as `prep` just repeatedely executes `func` for `n_iter` operations with optional args/kwargs that are the same for each run of `func`.

    Args:
        func (callable): function to run
        func_args (list/dict/None): arguments to the function provided as a list for unnamed args or a dict for named kwargs. If None, assumes func takes no arguments excepted loop_idx_available (if its True); Default None
        n_iter (int, optional): number of iterations; Default 100
        n_jobs (int, optional): number of cpus/threads; Default -1 (all cpus/threads)
        loop_idx (bool, optional): whether the value of the current iteration should be passed to func as the special kwarg 'idx'. Make sure func can handle a kwarg named 'idx'. Default True
        loop_random_seed (bool, optional): whether a randomly initialized seed should be passed to func as the special kwarg 'seed'. If func depends on any randomization (e.g. np.random) this should be set to True to ensure that parallel processes/threads use independent random seeds. Make sure func can handle a kwarg named 'seed' and utilize it for randomization. See example. Default False.
        backend (str, optional): 'processes' or 'threads'. Use 'threads' when you know you function releases Python's Global Interpreter Lock (GIL); Default 'cpus'
        progress (bool): whether to show a tqdm progress bar note, this may be a bit inaccurate when n_jobs > 1. Default True.
        verbose (int): joblib.Parallel verbosity. Default 0
        seed (int/None): random seed for reproducibility

    Examples:
        How to use a random seed.

        >>> from utilz.ops import prep, random_seed

        First make sure your function handles a 'seed' keyword argument. Then initialize it with the utilz.ops.random_seed function. Finally, use it internally where you would normally make a call to np.random.

        >>> def boot_sum(arr, seed=None):
        >>>     "Sum up elements of array after resampling with replacement"
        >>>     new_seed = random_seed(seed)
        >>>     boot_arr = new_seed.choice(arr, len(arr), replace=True)
        >>>     return boot_arr.sum()

        Finally call it in a parallel fashion

        >>> prep(boot_sum, [np.arange(10)], n_iter=100, loop_random_seed=True, loop_idx=False)
    """

    if backend not in ["processes", "threads"]:
        raise ValueError("backend must be one of cpu's threads")

    parfor = Parallel(prefer=backend, n_jobs=n_jobs, verbose=verbose)
    if loop_random_seed:
        seeds = random_seed(seed).randint(MAX_INT, size=n_iter)

    if progress:
        iterator = tqdm(range(n_iter))
    else:
        iterator = range(n_iter)

    if func_args is None:
        if loop_idx:
            if loop_random_seed:
                out = parfor(
                    delayed(func)(**{"idx": i, "seed": seeds[i]}) for i in iterator
                )
            else:
                out = parfor(delayed(func)(**{"idx": i}) for i in iterator)
        else:
            if loop_random_seed:
                out = parfor(delayed(func)(**{"seed": seeds[i]}) for i in iterator)
            else:
                out = parfor(delayed(func) for _ in iterator)
    else:
        if loop_idx:
            if loop_random_seed:
                if isinstance(func_args, list):
                    out = parfor(
                        delayed(func)(*func_args, **{"idx": i, "seed": seeds[i]})
                        for i in iterator
                    )
                elif isinstance(func_args, dict):
                    out = parfor(
                        delayed(func)(**func_args, **{"idx": i, "seed": seeds[i]})
                        for i in iterator
                    )
                else:
                    raise TypeError("func_args must be a list or dict")
            else:
                if isinstance(func_args, list):
                    out = parfor(
                        delayed(func)(*func_args, **{"idx": i}) for i in iterator
                    )
                elif isinstance(func_args, dict):
                    out = parfor(
                        delayed(func)(**func_args, **{"idx": i}) for i in iterator
                    )
                else:
                    raise TypeError("func_args must be a list or dict")
        else:
            if loop_random_seed:
                if isinstance(func_args, list):
                    out = parfor(
                        delayed(func)(*func_args, **{"seed": seeds[i]})
                        for i in iterator
                    )
                elif isinstance(func_args, dict):
                    out = parfor(
                        delayed(func)(**func_args, **{"seed": seeds[i]})
                        for i in iterator
                    )
                else:
                    raise TypeError("func_args must be a list or dict")
            else:
                if isinstance(func_args, list):
                    out = parfor(delayed(func)(*func_args) for _ in iterator)
                elif isinstance(func_args, dict):
                    out = parfor(delayed(func)(**func_args) for _ in iterator)
                else:
                    raise TypeError("func_args must be a list or dict")
    return out


# TODO: test me
def splitdf(df, X=None, Y=None):
    """
    Split a dataframe into X and Y arrays given column names. Useful for splitting up a a pandas dataframe as a `sklearn` pipeline step. If `Y` is `None`, assumes its in the *first* column of `df`.

    Args:
        df (Dataframe): input dataframe that's at least 2d
        Y (string, optional): name(s) of the columns for the Y array. Defaults to df.iloc[:, 0]
        X (string/list, optional): name(s) of columns for the X array. Defaults to df.iloc[:,1:]

    Raises:
        ValueError: If df is not 2 dimensional

    Returns:
        tuple: (X, Y)
    """

    if df.ndim != 2:
        raise ValueError("df must 2 dimensional")

    if X is not None:
        x = df.loc[:, X].to_numpy(copy=True)
    else:
        x = df.iloc[:, 1:].to_numpy(copy=True)

    if Y is not None:
        y = df.loc[:, Y].to_numpy(copy=True)
    else:
        y = df.iloc[:, 0].to_numpy(copy=True)

    return x, y


# TODO: test me
def mapcat(func, iterme, as_df=False, as_arr=False, axis=0, ignore_index=True):
    """
    **map**s `func` to `iterme` and con**cat**enates the result into a single, list, DataFrame or array. `iterme` can be a list of elements, list of DataFrames, list of arrays, or list of lists. List of lists up to 2 deep will be flattened to single list.

    A a few interesting use cases include:

    - Passing None for the value of func acts as a shortcut to flatten nested lists.
    - Using in place of `map` acts like a call to `list(map(...))`
    - Passing in `pd.read_csv` to list of files to get back a single dataframe

    Args:
        func (callable): function to apply. If None, will attempt to flatten a nested list
        iterme (iterable): an iterable for which func will be called on each element
        as_df (bool; optional): combine result into a DataFrame; Default False
        as_arr (bool; optional): combine result into an array; Default False
        axis (int; optional): what axis to concatenate over; Default 0 (first)
        ignore_index (bool; optional): ignore the index when combining DataFrames; Default True
    """

    if as_df and as_arr:
        raise ValueError("as_df and as_arr cannot both be True")

    if func is None:
        return list(chain.from_iterable(iterme))

    if isinstance(iterme[0], list):
        op = map(func, chain.from_iterable(iterme))
    else:
        op = map(func, iterme)

    op = list(op)

    if as_df:
        op = pd.concat(op, axis=axis, ignore_index=ignore_index)
    elif as_arr:
        try:
            op = np.concatenate(op, axis=axis)
        except np.AxisError as e:  # noqa
            warn(
                "Created new axis because requested concatenation axis > existing axes"
            )
            op = np.stack(op, axis=axis - 1)

    return op
