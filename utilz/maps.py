"""
The maps module is a generalization of many of the functions in `utilz.ops` that operate
on **iterables**. Here are the parallels:

| map function (s)   | op function(s)  | description |
|---|---|---|
| `map`  | `do`  | apply **one function** |
| `mapcompose`  | `pipe`/`do(compose())`  | apply **multiple functions in sequence** |
| `mapmany`  | `many`  | apply **multiple functions in parallel** |
| `mapacross`  | `None`  | apply **multiple functions** to **multiple inputs** in pairs
| `mapif`  | `iffy`  | apply **one function** if a *predicate function* otherwise noop |
| `mapcat`  | `None`/`concat`  | apply **one multi-output function** and flatten the results |
| `mapwith`  | `None`  | map a two argument function to an iterable and a fixed arg or two iterables |


All members of the `map` family, expect an iterable as their **last** argument, each
element of which is passed to functions as their **first** argument. Except for
`mapcat`, all `map*` functions return a sequence the same length as the input they received.
"""

__all__ = [
    "filter",
    "map",
    "mapcat",
    "mapcompose",
    "mapmany",
    "mapacross",
    "mapif",
    "mapwith",
]

from joblib import delayed, Parallel
from collections.abc import Callable, Iterable
from typing import Union, Any
from .ops import curry, concat, check_random_state, iffy, compose, many
from ._utils import ProgressParallel
from tqdm import tqdm
from inspect import signature
import numpy as np
from itertools import filterfalse
from copy import deepcopy


MAX_INT = np.iinfo(np.int32).max
_filter = filter  # we're overwriting the base func


# Helper used by map
def _pmap(
    func: Callable,
    iterme: Iterable,
    enum: bool = False,
    seeds: Union[None, list] = None,
    n_jobs: int = 1,
    backend: Union[None, str] = None,
    progressbar: bool = True,
    verbose: int = 0,
    func_kwargs: Union[None, dict] = None,
) -> Any:
    # Setup progress bars and parallelization
    if n_jobs < 1 or n_jobs > 1:
        # Initialize joblib parallelizer
        if progressbar:
            parfor = ProgressParallel(
                prefer=backend, n_jobs=n_jobs, verbose=verbose, total=len(iterme)
            )
        else:
            parfor = Parallel(prefer=backend, n_jobs=n_jobs, verbose=verbose)

        wrapped = delayed(func)
    # n_jobs == 1 so we loop normally to avoid overhead cost incurred from Parallel with
    # 1 job
    else:
        if progressbar:
            iterme = tqdm(iterme)

        wrapped = func

    # Without enumeration
    if not enum:
        # Without random seeds
        if seeds is None:
            iterator = iterme
            if func_kwargs is None:
                call_list = [wrapped(elem) for elem in iterator]
            else:
                call_list = [wrapped(elem, **func_kwargs) for elem in iterator]

        # With random seeds
        else:
            iterator = zip(iterme, seeds)
            if func_kwargs is None:
                call_list = [
                    wrapped(elem, random_state=seed) for elem, seed in iterator
                ]
            else:
                call_list = [
                    wrapped(elem, random_state=seed, **func_kwargs)
                    for elem, seed in iterator
                ]

    # With enumeration
    else:
        # Without random seeds
        if seeds is None:
            iterator = enumerate(iterme)
            if func_kwargs is None:
                call_list = [wrapped(elem, idx=i) for i, elem in iterator]
            else:
                call_list = [
                    wrapped(elem, idx=i, **func_kwargs) for i, elem in iterator
                ]

        # With random seeds
        else:
            iterator = enumerate(zip(iterme, seeds))
            if func_kwargs is None:
                call_list = [
                    wrapped(elem, idx=i, random_state=seed)
                    for i, (elem, seed) in iterator
                ]
            else:
                call_list = [
                    wrapped(elem, idx=i, random_state=seed, **func_kwargs)
                    for i, (elem, seed) in iterator
                ]

    if n_jobs < 1 or n_jobs > 1:
        return parfor(call_list)
    else:
        return call_list


@curry
def map(
    func: Union[Callable, None],
    iterme: Iterable,
    **kwargs,
):
    """
    Super-power your `for` loops with a progress-bar and optional *reproducible*
    parallelization!

    **map**s `func` to `iterme`. Includes a progress-bar powered by `tqdm`.

    Supports parallelization with `jobllib.Parallel` multi-processing by setting `n_jobs > 1`. Progress-bar *accurately* tracks parallel jobs!

    `iterme` can be a list of elements, list of DataFrames, list of arrays, or list of
    lists. List of lists up to 2 deep will be flattened to single list when `func = None`

    See the examples below for interesting use cases beyond standard looping!

    Args:
        func (callable): function to map
        iterme (iterable): an iterable for which each element will be passed to func
        into a single list, array, or dataframe based on `axis`; Default True
        enum (bool, optional): whether the value of the current iteration should be passed to `func` as the special kwarg `idx`. Make sure `func` can handle a kwarg named `idx`. Default False
        random_state (bool/int, optional): whether a randomly initialized seed should be
        passed to `func` as the special kwarg `random_state`. The function should pass
        this seed to the `utilz.check_random_state` helper to generate a random number
        generator for all computations rather than relying on `np.random`
        n_jobs (int, optional): number of cpus/threads; Default 1 (no parallel)
        backend (str, optional): Only applies if `n_jobs > 1`. See `joblib.Parallel` for
        options; Default None which uses `loky`
        Default True
        pbar (bool, optional): whether to use tqdm to sfunc a progressbar; Default
        False
        verbose (int): `joblib.Parallel` verbosity. Default 0
        **kwargs (dict, optional): optional keyword arguments to pass to func

    Examples:
        >>> # Just like map
        >>>  out = map(lambda x: x * 2, [1, 2, 3, 4])

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

    enum = kwargs.pop("enum", False)
    random_state = kwargs.pop("random_state", False)
    n_jobs = kwargs.pop("n_jobs", 1)
    backend = kwargs.pop("backend", None)
    pbar = kwargs.pop("pbar", False)
    verbose = kwargs.pop("verbose", 0)

    if func is None:
        # No-op if no function
        op = iterme
    else:
        if isinstance(func, (str, dict, int, float, tuple, dict)):
            func_args = []
        else:
            try:
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
            except ValueError as _:
                # some funcs like numpy c funcs are not inspectable so we have to ksip
                # these checks
                func_args = []

        if random_state is not False:
            # User can pass True instead of a number for non-reproducible
            # parallelization
            random_state = None if random_state is True else random_state

            # Generate a list of random ints, that themselves are seeded by random_state
            # and passed to func
            seeds = check_random_state(random_state).randint(MAX_INT, size=len(iterme))
        else:
            seeds = None

        # Loop; parallel in n_jobs < 1 or > 1
        op = _pmap(func, iterme, enum, seeds, n_jobs, backend, pbar, verbose, kwargs)

    return op


@curry
def mapcat(func: Union[Callable, None], iterme: Iterable, **kwargs):
    """Call map and concatenate results after.
    Particularly useful to ensure results are numpy arrays"""

    concat_axis = kwargs.pop("concat_axis", None)
    ignore_index = kwargs.pop("ignore_index", True)
    out = map(func, iterme, **kwargs)

    return concat(out, iterme, concat_axis, ignore_index)


@curry
def mapacross(*args):
    """Map multiple functions to an iterable in a matched-pair fasion. The
    number of funcions needs to equal the length of the iterable."""

    def call(data):
        if not isinstance(data, (list, tuple)):
            raise TypeError(
                f"Expected a list/tuple of input, but received a single {type(data)}. If you want to apply a function to a single input either use a lambda or do()"
            )
        if len(data) != len(args):
            raise ValueError(
                f"Te number of functions passed must equal the length of the previous output, but {len(data)} data and {len(args)} functions don't match. To run the same set of functions over the previous inputs see separate()"
            )
        return [f(a) for f, a in zip(args, data)]

    return call


@curry
def mapmany(*args, **kwargs):
    """Map multiple functions separately to each element in an iterable. Returns a list
     of nested lists containing the output of each function evaluation on each element in
    iterme"""

    def call(data):
        if not isinstance(data, (list, tuple)):
            raise TypeError(
                f"All map* funcs expect a list/tuple of input, but received a single {type(data)}."
            )
        if len(args) <= 1:
            raise ValueError(
                f"mapmany applies *multiple* function calls separately but only received {len(args)} function. Use mapcat() to apply a single function."
            )

        together = many(*args)
        return map(together, data, **kwargs)

    return call


@curry
def mapcompose(*args, **kwargs):
    """Compose multiple functions together and map them over a sequences, i.e. a
    mini-pipe per element. Returns a list the same length as the input iterable
    containing the final function evaluation for each element."""

    def call(data):
        if not isinstance(data, (list, tuple)):
            raise TypeError(
                f"All map* funcs expect a list/tuple of input, but received a single {type(data)}."
            )
        if len(args) <= 1:
            raise ValueError(
                f"mapcompose applies *multiple* function calls in sequence but only received {len(args)} function. Use mapcat() to apply a single function."
            )

        composed = compose(*args)
        return map(composed, data, **kwargs)

    return call


@curry
def mapif(func, predicate_func, iterme, **kwargs):
    """Apply func to each element of iterme if predicate_func is True for that element
    otherwise return the element"""

    return map(iffy(predicate_func, func), iterme, **kwargs)


@curry
def filter(
    how: Union[Callable, Iterable, str, int, float],
    iterme: Iterable,
    invert: Union[str, bool] = False,
    substr_match: bool = True,
    assert_notempty: bool = True,
):
    """
    Filter an iterable and concatenate the output to a list instead of a generator like
    the standard `filter` in python. By default always returns the *matching* elements
    from iterme. This can be inverted using invert=True or split using invert='split'
    which will return matches, nomatches. Filtering can be done by passing a function, a
    single `str/int/float`, or an iterable of `str/int/float`. Filtering by an iterable
    checks if `any` of the values in the iterable exist in each item of `iterme`.

    Args:
        func (Union[Callable, Iterable, str, int, float]): if a function is passed it
        must return `True` or `False`, otherwise will compare each element in `iterme`
        to the element passed for `func`. String comparisons check if `func` is `in` and
        element of `iterme` while float/integer comparisons check for value equality. If
        an iterable is passed filtering is performed for `any` of the elements in the ierable
        iterme (Iterable): iterable to filter
        invert (bool/str optional): if `True`, drops items where `how` resolves to
        `True` rather than keeping them. If passed the string `'split'` will return both
        matching and inverted results
        assert_notempty (bool, optional): raise an error if the returned output is
        empty; Default True


    Returns:
        list: filtered version of `iterme
    """

    if isinstance(how, Callable):
        func = how
    elif isinstance(how, str):
        if substr_match:
            func = lambda elem: how in str(elem)
        else:
            func = lambda elem: how == elem
    elif isinstance(how, (float, int)):
        func = lambda elem: how == elem
    elif isinstance(how, Iterable):
        if isinstance(how[0], str):
            if substr_match:
                func = lambda elem: any(map(lambda h: h in str(elem), how))
            else:
                func = lambda elem: any(map(lambda h: h == elem, how))
        elif isinstance(how[1], (float, int)):
            func = lambda elem: any(map(lambda h: h == elem, how))
        else:
            raise TypeError(
                "If an iterable is passed it must contain strings, ints or floats"
            )
    else:
        raise TypeError(
            "Must pass a function, iterable, string, int, or float to filter by"
        )

    if invert == "split":
        inverts = list(filterfalse(func, iterme))
        matches = list(_filter(func, iterme))

        if assert_notempty and (len(inverts) == 0 or len(matches) == 0):
            raise AssertionError("Filtered data is empty!")
        return matches, inverts

    elif isinstance(invert, bool):
        filtfunc = filterfalse if invert is True else _filter
        out = list(filtfunc(func, iterme))
        if assert_notempty and len(out) == 0:
            raise AssertionError("Filtered data is empty!")
        return out
    else:
        raise TypeError("invert must be True, False, or 'split'")


@curry
def mapwith(func, iterwith, iterme, **kwargs):
    """Just like map but accepts a second arg that can also be an iterator. In a pipe
    iterme is always the *last* input to mapwith, but the *first* input func. If
    `copy=True` is passed and iterwith is not an iterator, an iterator is built with
    guaranteed copies of iterwith."""

    copy = kwargs.pop("copy", False)

    if not isinstance(iterwith, (list, tuple)):
        if copy:
            hascopy = getattr(iterwith, "copy", None)
            if callable(hascopy):
                iterwith = [iterwith.copy()] * len(iterme)
            else:
                iterwith = [deepcopy(iterwith)] * len(iterme)
        else:
            iterwith = [iterwith] * len(iterme)

    if len(iterme) != len(iterwith):
        raise TypeError(
            f"mapwith received an iterable but its length ({len(iterwith)} doesn't match the length of the input iterable ({len(iterme)}"
        )

    return map(lambda tup: func(*tup), zip(iterme, iterwith), **kwargs)
