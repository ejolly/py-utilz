"""
Common data operations and transformations. Often on pandas dataframes
"""

from cytoolz import memoize, curry
from joblib import Parallel, delayed

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
    backend="processes",
    progress=True,
):
    """
    Call a function for n_iter using parallelization via joblib

    Args:
        func (callable): function to run
        func_args (list/dict/None): arguments to the function provided as a list for unnamed args or a dict for named kwargs. If None, assumes func takes no arguments excepted loop_idx_available (if its True); Default None
        n_iter (int, optional): number of iterations; Default 100
        n_jobs (int, optional): number of cpus/threads; Default -1 (all cpus/threads)
        loop_idx_available (bool, optional): whether the value of the current iteration should be passed as the last argument to func; Default True
        backend (str, optional): 'processes' or 'threads'. Use 'threads' when you know you function releases Python's Global Interpreter Lock (GIL); Default 'cpus'
        progress (bool, opional): whether to show progress; Default True
    """

    if backend not in ["processes", "threads"]:
        raise ValueError("backend must be one of cpu's threads")

    verbose = 10 if progress else 0
    parfor = Parallel(prefer=backend, n_jobs=n_jobs, verbose=verbose)

    if func_args is None:
        if loop_idx_available:
            out = parfor(delayed(func)(i) for i in range(n_iter))
        else:
            out = parfor(delayed(func) for i in range(n_iter))
    else:
        if loop_idx_available:
            out = parfor(delayed(func)(*func_args, i) for i in range(n_iter))
        else:
            out = parfor(delayed(func)(*func_args) for i in range(n_iter))

    return out

