"""
Functional tools

---
"""
__all__ = [
    "check_random_state",
    "mapcat",
    "filtercat",
    "sort",
    "pipe",
    "append",
    "spread",
    "separate",
    "gather",
    "do",
    "ifelse",
    "compose",
    "curry",
]

from joblib import delayed, Parallel
from ._utils import ProgressParallel
import numpy as np
import pandas as pd
from typing import Union, Any
from collections.abc import Callable, Iterable
from itertools import chain, filterfalse
from inspect import signature
from tqdm import tqdm
from toolz import curry, juxt
from toolz.curried import compose_left as compose
from matplotlib.figure import Figure, Axes
from matplotlib.axes._subplots import Subplot
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
    pbar: bool = False,
    verbose: int = 0,
    func_kwargs: Union[None, dict] = None,
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
        pbar (bool, optional): whether to use tqdm to sfunc a progressbar; Default
        False
        verbose (int): `joblib.Parallel` verbosity. Default 0
        func_kwargs (dict, optional): optional keyword arguments to pass to func;
        Default None

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

    if func_kwargs is not None and not isinstance(func_kwargs, dict):
        raise TypeError(
            "func_kwargs should be pass as a dictionary of kwarg: value names, like: func_kwargs={'ddof': 2}"
        )
    if func is None:
        # No-op if no function
        op = iterme
    else:
        if (
            func is str
            or func is dict
            or func is int
            or func is float
            or func is tuple
            or func is dict
        ):
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
        op = _pmap(
            func, iterme, enum, seeds, n_jobs, backend, pbar, verbose, func_kwargs
        )

    if concat:
        return _concat(op, iterme, axis, ignore_index)
    return op


# Helper used by mapcat
def _concat(op, iterme, axis, ignore_index):

    try:
        if isinstance(op[0], (pd.DataFrame, pd.Series)):
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
        if isinstance(op[0], list):
            return list(chain.from_iterable(op))
        return op

    except Exception as e:
        print(e)
        return op


@curry
def filtercat(
    how: Union[Callable, Iterable, str, int, float],
    iterme: Iterable,
    invert: Union[str, bool] = False,
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
        func = lambda elem: how in str(elem)
    elif isinstance(how, (float, int)):
        func = lambda elem: how == elem
    elif isinstance(how, Iterable):
        if isinstance(how[0], str):
            func = lambda elem: any(map(lambda h: h in str(elem), how))
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
        matches = list(filter(func, iterme))

        if assert_notempty and (len(inverts) == 0 or len(matches) == 0):
            raise AssertionError("Filtered data is empty!")
        return matches, inverts

    elif isinstance(invert, bool):
        filtfunc = filterfalse if invert is True else filter
        out = list(filtfunc(func, iterme))
        if assert_notempty and len(out) == 0:
            raise AssertionError("Filtered data is empty!")
        return out
    else:
        raise TypeError("invert must be True, False, or 'split'")


@curry
def sort(iterme: Iterable, **kwargs):
    return sorted(iterme, **kwargs)


def pipe(data: Any, *funcs: Iterable, output: bool = True):
    """
    A "smart" pipe function designed to pass data through a series of transformation.
    Similar into `toolz.pipe` in that it performs a series of nested function
    evaluations. But it always *displays* the last function evaluation, even when
    assigning to a variable, making it useful when working in an interactive environment
    or logging from a script. Also recognizes if the last function evaluation returns
    None or a plot and returns last non-None/non-plot evaluation in the pipe. Passing
    output = False will return nothing from the pipe, which is if you just want to
    run a pipe for its side-effects, e.g. saving a figure, looking at something
    """

    if not funcs:
        return data

    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            from IPython.display import display as printfunc
        else:
            printfunc = print
    except NameError:
        printfunc = print

    show_last_eval = True
    plot_types = (Figure, Axes, Subplot)
    isorhasplot = lambda e: isinstance(e, plot_types) or (
        isinstance(e, tuple) and isinstance(e[0], plot_types)
    )

    evals = []
    orig = data
    out = None
    for f in funcs:
        data = f(data)
        evals.append(data)

    if isorhasplot(evals[-1]):
        show_last_eval = False

    for e in evals[::-1]:
        if e is None:
            continue
        elif isorhasplot(e):
            continue
        else:
            out = e
            break

    if out is None:
        out = orig
    if show_last_eval:
        printfunc(out)
    if output:
        return out


@curry
def spread(*args):
    """Takes multiple functions and applies each to the input, returning a tuple the
    same length as args"""

    from copy import deepcopy

    if len(args) == 1:
        if isinstance(args[0], int):

            def duplicate(data):
                copy = getattr(data, "copy", None)
                if callable(copy):
                    return tuple([data.copy()] * args[0])
                return tuple([deepcopy(data)] * args[0])

            return duplicate
        else:
            raise ValueError(
                f"only 1 function passed to spread. Use do() instead or simply call the function directly in the pipe"
            )

    elif all(callable(a) for a in args):
        together = juxt(*args)
        return together
    else:
        raise TypeError(f"spread expected an integer or 1+ functions")


@curry
def append(func):
    """Takes a function and returns a new function that prepends the args to the
    function as part of the input, i.e. (input, funcval)"""

    def alongwith(data):
        # If data is a tuple and func only takes 1 arg, then assume the user wants the
        # original data in the chain
        if isinstance(data, tuple):
            sig = signature(func)
            if len(sig.parameters) == 1:
                out = func(data[0])
            else:
                # Otherwise give them the entire chain
                out = func(*data)
            return (*data, out)
        else:
            out = func(data)
            return (data, out)

    return alongwith


@curry
def gather(func, data):
    """Takes a single function and a tuple of data and unpacks the tuple as multiple
    arguments to the function. Useful after a call to append, spread, separate, or any
    function that returns more than 1 output
    """

    if not (isinstance(data, (list, tuple)) and len(data) > 1):
        raise TypeError(
            f"gather expects the previous step's output to be a list/tuple of length > 1 but received a {type(data)}"
        )

    return func(*data)


@curry
def separate(*args, match=False):
    """Apply one or more functions to multiple inputs separately. If the number of
    functions is greater or less than the number of inputs, then each input will be run
    through all functions in a sequence (like a mini-pipe). If the number of functions
    == the number of inputs and match=True, then each input-function pair will be
    evaluated separately."""

    def call(data):
        if not isinstance(data, (list, tuple)):
            raise TypeError(
                f"Expected a list/tuple of input, but received a single {type(data)}. If you want to apply a function to a single input either use a lambda or do()"
            )
        # We apply each function to each data
        if match:
            if len(data) != len(args):
                raise ValueError(
                    f"To use match=True, the number of functions passed must equal the length of the previous output, but {len(data)} data and {len(args)} functions don't match"
                )
            return tuple([f(a) for f, a in zip(args, data)])
        else:
            out = []
            for d in data:
                for func in args:
                    d = func(d)
                out.append(d)
            return tuple(out)

    return call


@curry
def do(func, data, *args, **kwargs):
    """Apply a single function to data or call a method on data, while passing optional
    kwargs to that functinon or method"""
    from operator import methodcaller as mc

    if isinstance(func, str):
        func = mc(func, *args, **kwargs)
    return func(data)


@curry
def ifelse(conditional, if_true, if_false, *args, **kwargs):
    """
    Simple oneline ternary operator. Pass in something to check, how to check it, what
    to return if True, and what to return if False. If only one of if_true or if_false
    is provided, then data is returned if the conditional matches that outcome. This
    makes it easy to run shorthands like "do something only if this is true"

    Args:
        data (any): thing to check
        conditional (bool or callable): a boolean expression or callable that will be
        applied to data
        if_true (None, any, optional): None, object, or callable. Defaults to None.
        if_false (None, any, optional): None, object, or callable. Defaults to None.

    Returns:
        Any: return from if_true or if_false
    """

    if len(args) == 0:
        data = None
    else:
        data = args[0] if len(args) == 1 else args

    if data is None:

        if if_true is None or if_false is None:
            raise ValueError(
                "when data is None, at least one of if_true or if_false must be provided"
            )

        if callable(conditional):
            conditional = conditional(data)

        if isinstance(conditional, str):
            conditional = eval(conditional)

        return if_true if conditional else if_false

    if callable(conditional):
        conditional = conditional(data)
    if isinstance(conditional, str):
        conditional = eval(conditional)

    if conditional:
        if callable(if_true):
            return if_true(data, **kwargs)
        if if_true is None:
            return data
        return if_true
    else:
        if callable(if_false):
            return if_false(data, **kwargs)
        if if_false is None:
            return data
        return if_false
