"""
Common data operations and transformations. Often on pandas dataframes
"""

from cytoolz import memoize, curry
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np

MAX_INT = np.iinfo(np.int32).max


def random_seed(seed):
    """Turn seed into a np.random.RandomState instance. Note: credit for this code goes entirely to sklearn.utils.check_random_state. Using the source here simply avoids an unecessary dependency.

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
    loop_idx_available=True,
    randomize=False,
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
        loop_idx_available (bool, optional): whether the value of the current iteration should be passed as the last argumentto fun. Make sure func expects a integer as its last arg; Default True
        randomize (bool, optional): if func depends on any randomization (e.g. np.random) this should be set to True to ensure that parallel processes/threads use independent random seeds. func should take a keyword argument 'seed' and use it internally. See example. Default False.
        backend (str, optional): 'processes' or 'threads'. Use 'threads' when you know you function releases Python's Global Interpreter Lock (GIL); Default 'cpus'
        progress (bool): whether to show a tqdm progress bar note, this may be a bit inaccurate when n_jobs > 1. Default True.
        verbose (int): joblib.Parallel verbosity. Default 0
        seed (int/None): random seed for reproducibility

    Examples:
        How to use a random seed.
        
        First make sure your function takes a random seed as input and uses the utility function random_seed to perform randomization.
        
        >>> from utilz.ops import ploop, random_seed
        
        >>> def boot_sum(arr, seed):
        >>>     "Sum up elements of array after resampling
        >>>     new_seed = random_seed(seed)
        >>>     boot_arr = new_seed.choice(arr, len(arr), replace=True)
        >>>     return boot_arr.sum()
        
        Then call it in a parallel fashion
        
        >>> ploop(boot_sum, [np.arange(10)], n_iter=100, randomize=True)
    """

    if backend not in ["processes", "threads"]:
        raise ValueError("backend must be one of cpu's threads")

    parfor = Parallel(prefer=backend, n_jobs=n_jobs, verbose=verbose)
    if randomize:
        seeds = random_seed(seed).randint(MAX_INT, size=n_iter)

    if progress:
        iterator = tqdm(range(n_iter))
    else:
        iterator = range(n_iter)

    if func_args is None:
        if loop_idx_available:
            if randomize:
                out = parfor(delayed(func)(i, {"seed": seeds[i]}) for i in iterator)
            else:
                out = parfor(delayed(func)(i) for i in iterator)
        else:
            if randomize:
                out = parfor(delayed(func)({"seed": seeds[i]}) for i in iterator)
            else:
                out = parfor(delayed(func) for i in iterator)
    else:
        if loop_idx_available:
            if randomize:
                out = parfor(
                    delayed(func)(*func_args, i, {"seed": seeds[i]}) for i in iterator
                )
            else:
                out = parfor(delayed(func)(*func_args, i) for i in iterator)
        else:
            if randomize:
                out = parfor(
                    delayed(func)(*func_args, {"seed": seeds[i]}) for i in iterator
                )
            out = parfor(delayed(func)(*func_args) for i in iterator)

    return out


