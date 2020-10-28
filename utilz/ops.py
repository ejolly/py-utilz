"""
Common data operations and transformations often on pandas dataframes

---
"""

from cytoolz import memoize, curry
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np

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


@curry
@memoize
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


def ploop(
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
    Call a function for n_iter using parallelization via joblib

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

        >>> from utilz.ops import ploop, random_seed

        First make sure your function handles a 'seed' keyword argument. Then initialize it with the utilz.ops.random_seed function. Finally, use it internally where you would normally make a call to np.random.

        >>> def boot_sum(arr, seed=None):
        >>>     "Sum up elements of array after resampling with replacement"
        >>>     new_seed = random_seed(seed)
        >>>     boot_arr = new_seed.choice(arr, len(arr), replace=True)
        >>>     return boot_arr.sum()

        Finally call it in a parallel fashion

        >>> ploop(boot_sum, [np.arange(10)], n_iter=100, loop_random_seed=True, loop_idx=False)
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
