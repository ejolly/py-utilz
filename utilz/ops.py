"""
Functional tools intended to be used with `pipe()`. Everything in this module except for
`pipe` itself, is *curried* so can be called without a full set of args.

## Overview

| function (s)   | description  | 
|---|---|
| `do`  | apply a function or method to an object |
| `pipe`  | run an input through a sequence of functions  |
| `append`/`alongwith`  | apply a function and return `(input, result)` as a `tuple` |
| `fork`  | call `input.copy` if possible otheriwse create `n` duplicate `deepcopy`'s of `input` |
| `many`  | applies **multiple functions in parallel** to a **single** input and returns an iterable; like an "inverse `map`" |
| `spread`  | acts like `fork` if given an `int` otherwise acts like `many` |
| `across`  | apply **multiple functions** to **multiple inputs** in pairs; alias for `mapacross` |
| `compose`  | combine **multiple functions** into **one function** sequence (mini-pipe) |
| `gather`/`unpack`  | make an iterable's items separately accessible to a **single function** |
| `iffy`  | apply a function if a predicate function is true otherwise noop |

---
"""
__all__ = [
    "pipe",
    "append",
    "alongwith",
    "across",
    "sort",
    "fork",
    "many",
    "spread",
    "gather",
    "unpack",
    "do",
    "iffy",
    "compose",
    "curry",
    "pop",
    "nth",
    "check_random_state",
    "pairs",
]

import numpy as np
import pandas as pd
from typing import Union, Any
from collections.abc import Callable, Iterable
from itertools import chain as it_chain
import itertools as it
from inspect import signature
from toolz import curry, juxt
from toolz.curried import compose_left as compose, nth
from inspect import signature
from pathlib import Path


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


def concat(op, iterme, axis, ignore_index):
    """Intelligently try to concatenate an iterable. Supports dataframe, arrays, and lists"""

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
            return list(it_chain.from_iterable(op))
        return op

    except Exception as e:
        print(e)
        return op


@curry
def sort(iterme: Iterable, **kwargs):
    return sorted(iterme, **kwargs)


@curry
def pairs(iterme: Iterable):
    return list(it.combinations(iterme, 2))


def pipe(
    data: Any,
    *funcs: Iterable,
    output: bool = True,
    show: bool = False,
    debug: bool = False,
    keep: Union[int, None] = None,
    load_existing: bool = False,
    save: Union[list, Path, str, bool, None] = False,
    flatten: bool = False,
):
    """
    A "smart" pipe function designed to pass data through a series of transformation.
    Accepts an initial object "data" and then inumerable additional `args` which are
    functions run in sequence: `data -> f1(data) -> f2(f1_output) -> f3(f2_output)...`.

    Using `load_existing` with `save` can fully bypass the evaluation of a pipe if a
    file already exists on disk.

    Use `show=True` to *display* the last function evaluation, even when
    assigning to a variable, making it useful when working in an interactive environment
    or logging from a script.

    Passing `output = False` will return nothing from the pipe, which is useful if you
    just want to run a pipe for its side-effects, e.g. saving a figure, looking at
    something.

    pipe supports `...` as a **special semantic** to denote what to return.
    Everything *before* `...` will be evaluated and returned, while everything *after*
    `...` will be evaluated by *not* returned. For example in `out = pipe(data, f1, f2,
    ..., f3, f4)` only the output up until `...` will be stored in `out`, so
    `f2(f1(data))`. `f3` and `f4` will still run, but never return their outputs. For more details see [here](https://eshinjolly.com/utilz/pipes/#ellipses)

    Args:
        data (Any): input data
        output (bool, optional): whether to return a result. Defaults to True.
        show (bool, optional): whether to display the result. Defaults to True.
        debug (bool, optional): whether to return a list of all function evaluations. Defaults to False.
        keep (Union[int, None], optional): indices to slices in the input if the last
        step of the pipe returns multiple outputes. Defaults to None.
        load_existing (bool, optional): is `save` is not `False`, will try to load the
        path(s) provided to `save` thus bypassing the pipe. Defaults to False.
        save (Union[list, Path, str, bool, None], optional): one or more file paths to
        save the outputs of the pipe to. Defaults to False.

    """

    if not funcs:
        return data

    # For auto-displaying in interactive environments
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            from IPython.display import display as printfunc
        else:
            printfunc = print
    except NameError:
        printfunc = print

    # Keep track of function evaluations only during debugging to reduce memory
    if debug:
        evals = []
    out = None

    # load_existing guard
    if save and not debug:
        if not isinstance(save, list):
            save = [save]
        if load_existing:
            if all(map(lambda f: Path(f).exists(), save)):
                # Only support csvs for now
                print("Existing csv(s) found. Bypassing pipe and loading from disk...")
                out = tuple([pd.read_csv(s) for s in save])

    # Actually run pipe
    if out is None:
        for f in funcs:
            if f is Ellipsis:
                if out is None:
                    out = data
                    continue
                else:
                    raise ValueError("There can only be one ... inside a pipe")
            data = f(data)
            if debug:
                evals.append(data)

        if debug:
            return evals

    # If the pipe was passed an ..., then we don't need to run this block
    if out is None:
        out = data

    if show:
        if isinstance(out, tuple):
            for o in out:
                printfunc(o)
        else:
            printfunc(out)

    if output:
        # Single index output
        if isinstance(keep, int):
            out = out[keep]
        # Fancy index output
        elif isinstance(keep, (list, tuple, np.ndarray)):
            out = tuple([out[i] for i in keep])
        # Squeeze
        elif isinstance(out, tuple) and len(out) == 1:
            out = out[0]

        # Can only save things with output
        if save and not load_existing:
            if isinstance(out, pd.DataFrame):
                out.to_csv(save, index=False)
            elif isinstance(out, tuple):
                if not isinstance(save, list):
                    save = [save]

                for o, s in zip(out, save):
                    o.to_csv(s, index=False)

        if flatten and isinstance(out, tuple):
            _out = []
            for o in out:
                if isinstance(o, tuple):
                    _out.extend(list(o))
                else:
                    _out.append(o)
            out = tuple(_out)
        return out


@curry
def chain(*args, **kwargs):
    """Chain a sequence of functions on an input, i.e. a mini-pipe"""

    def call(data):
        for f in args:
            data = f(data, **kwargs)

    return call


@curry
def many(*args):
    """Apply many functions separately to a single input. Operates like the inverse of
    map(). Whereas map takes applies 1 func to multiple elements, many applies multiple funcs to 1 element. Returns a tuple the same length as args containing the output of each function
    """

    def call(data):
        if isinstance(data, (list, tuple)):
            raise TypeError(
                f"Expected a single input but receive {len(data)}. Use mapmany() to operate on an iterable"
            )
        if len(args) <= 1:
            raise ValueError(
                f"many applies *multiple* function calls separately but only received {len(args)} function. Use map() to apply a single function."
            )

        return tuple([f(data) for f in args])

    return call


@curry
def fork(*args):
    """Duplicate an input N number of times"""
    from copy import deepcopy

    def duplicate(data):
        copy = getattr(data, "copy", None)
        if callable(copy):
            return tuple([data.copy()] * args[0])
        return tuple([deepcopy(data)] * args[0])

    return duplicate


# Deprecate?
@curry
def spread(*args):
    """Generalization of fork OR many"""

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
    """Takes a function or obj and returns a new function that prepends the args to the
    function as part of the input, i.e. (input, funcval)"""

    def alongwith(data):
        # If data is a tuple and func only takes 1 arg, then assume the user wants the
        # original data in the chain
        if callable(func):
            if isinstance(data, tuple):
                sig = signature(func)
                if len(sig.parameters) == 1:
                    out = func(data[0])
                else:
                    out = func(*data)
                # Otherwise give them the entire chain
                return (*data, out)
            else:
                out = func(data)
                return (data, out)
        return (data, func)

    return alongwith


# Alias
@curry
def alongwith(thing):
    return append(thing)


# Alias for mapacross but returns tuple
def across(*args):
    """Like mapacross but returns a tuple"""

    def call(data):
        if not isinstance(data, (list, tuple)):
            raise TypeError(
                f"Expected a list/tuple of input, but received a single {type(data)}. If you want to apply a function to a single input either use a lambda or do()"
            )
        if len(data) != len(args):
            raise ValueError(
                f"Te number of functions passed must equal the length of the previous output, but {len(data)} data and {len(args)} functions don't match. To run the same set of functions over the previous inputs see separate()"
            )
        return tuple([f(a) for f, a in zip(args, data)])

    return call


@curry
def gather(func, data):
    if not (isinstance(data, (list, tuple)) and len(data) > 1):
        raise TypeError(
            f"gather expects the previous step's output to be a list/tuple of length > 1 but received a {type(data)}"
        )

    return func(*data)


# Alias for gather
@curry
def unpack(func, data):
    """Wraps a function that takes multiple inputs to make the output of a previous
    function with multiple outputs easier to work with. Useful after a call to `append`,
    `spread`, `across` or `mapmany` e.g.

    `unpack(lambda first_name, last_name: first_name + last_name)`
    """

    return gather(func, data)


@curry
def do(func, data, *args, **kwargs):
    """Apply a single function to data or call a method on data, while passing optional
    kwargs to that functinon or method"""
    from operator import methodcaller as mc

    if isinstance(func, str):
        func = mc(func, *args, **kwargs)
    return func(data)


@curry
def iffy(predicate_func: Callable, if_true: Union[Any, Callable], data: Any):
    """
    Conditionally apply a function based on a predicate function. Useful to
    conditionally map a function to an interable when combined with mapcat

    Args:
        conditional (callable): callable function on which data will be evaluated.
        Should return a boolean value
        if_true (any, callable, optional): Value or function to call if predicate_func
        is True
        data (any): previous pipe output

    """

    if not callable(predicate_func):
        raise TypeError("ifelse requires a function that returns a boolean value")

    if predicate_func(data):
        if callable(if_true):
            return if_true(data)
        else:
            return if_true
    else:
        return data


@curry
def pop(idx):
    """Given a tuple, removes an element located at an index. Useful for pruning down a
    call to append or spread."""

    if isinstance(idx, int):

        def remove(data):
            if isinstance(data, (tuple, list)):
                data = list(data) if isinstance(data, tuple) else data
                _ = data.pop(idx)
                out = tuple(data)
                out = out[0] if len(out) == 1 else out
                return out
            else:
                raise TypeError(
                    f"expected a tuple of input data by received a single {type(data)}"
                )

        return remove
    else:
        raise TypeError("pop requires an integer index to of the ouput to drop")
