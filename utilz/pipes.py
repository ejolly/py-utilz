"""
Functional tools for building data-transformation pipelines. There are two equivalent
ways to chain transformations:

```python
# Explicit pipe() call
pipe(data, f1, f2, f3)

# Unix-style | operator (pipe() is optional)
data | f1 | f2 | f3
```

Every helper in this module returns a `Pipe`: a thin wrapper around a one-argument
function that is both directly callable (so it works inside `pipe()`) and supports the
`|` operator via `__ror__` (so `data | helper(...)` runs `helper(...)(data)`). To make
your *own* function `|`-able, wrap it with `pipeable` — either as a decorator
(`@pipeable`) or inline (`data | pipeable(lambda x: x + 1)`).

## Overview

| function (s)   | description  |
|---|---|
| `pipe`  | run an input through a sequence of functions  |
| `pipeable`  | wrap a plain one-arg function/lambda so it supports the `\\|` operator |
| `Pipe`  | wrapper class that powers the `\\|` operator (returned by every helper) |
| `append`/`alongwith`  | apply a function and return `(input, result)` as a `tuple` |
| `fork`  | call `input.copy` if possible otheriwse create `n` duplicate `deepcopy`'s of `input` |
| `spread`  | acts like `fork` if given an `int` otherwise acts like `many` |
| `gather`/`unpack`  | make an iterable's items separately accessible to a **single function** |
| `pop`  | pop off an element from an iterable, useful for pruning down calls to `alongwith`/`spread` |

---
"""
__all__ = [
    "pipe",
    "Pipe",
    "pipeable",
    "append",
    "alongwith",
    "fork",
    "spread",
    "gather",
    "unpack",
    "pop",
]

import numpy as np
import pandas as pd
from functools import update_wrapper, wraps
from typing import Union, Any, Callable
from collections.abc import Iterable
from inspect import signature
from toolz import curry, juxt
from pathlib import Path


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


class Pipe:
    """Wrapper around a one-argument function that powers the `|` pipe operator.

    A `Pipe` is both directly callable and `|`-able, so the same object works in both
    pipelining styles:

        result = data | step      # runs step(data) via __ror__
        result = pipe(data, step)  # runs step(data) by calling it

    `Pipe`s also compose with each other point-free, letting you build a reusable
    pipeline and apply it later:

        cleanup = pipeable(strip) | pipeable(dropna)  # a new Pipe
        data | cleanup                                 # dropna(strip(data))

    The `|` operator relies on Python falling back to `Pipe.__ror__` when the left
    operand's own `__or__` returns `NotImplemented`. This works for plain Python
    objects (lists, dicts, tuples, ints) and `polars` DataFrames, but **not** for
    `numpy` arrays or `pandas` objects, whose `|` broadcasts element-wise. Start
    those chains with `pipe(...)` instead.

    Args:
        func (Callable): a one-argument function applied to the piped data.
    """

    def __init__(self, func: Callable):
        if not callable(func):
            raise TypeError(f"Pipe expects a callable, but got {type(func).__name__}")
        self.func = func
        # Preserve the wrapped function's name/docstring for nicer reprs and help()
        update_wrapper(self, func)

    def __call__(self, data: Any) -> Any:
        """Apply the wrapped function directly, e.g. inside `pipe()`."""
        return self.func(data)

    def __ror__(self, data: Any) -> Any:
        """Enable `data | self` syntax, equivalent to `self.func(data)`."""
        return self.func(data)

    def __or__(self, other: Callable) -> "Pipe":
        """Compose two steps point-free: `self | other` -> `other(self(x))`."""
        if callable(other):
            this, nxt = self.func, other
            return Pipe(lambda data: nxt(this(data)))
        return NotImplemented

    def __repr__(self) -> str:
        name = getattr(self.func, "__name__", repr(self.func))
        return f"Pipe({name})"


def _pipeify(verb: Callable) -> Callable:
    """Make a *verb factory* `|`-able by wrapping the transform it returns in a `Pipe`.

    Verbs in `utilz.dfverbs` are factories: calling `mutate(c="a + b")` returns a
    one-argument function `call(df)` that does the work. This wrapper turns that
    returned function into a `Pipe`, so `df | mutate(...)` works while
    `pipe(df, mutate(...))` keeps working unchanged.

    Verbs that return data directly (e.g. `read_csv`, `concat`) return a
    non-callable, which is passed through untouched.

    Args:
        verb (Callable): a verb factory to make pipe-operator aware.

    Returns:
        Callable: a wrapper that returns a `Pipe` when the verb returns a transform.
    """

    @wraps(verb)
    def wrapper(*args, **kwargs):
        result = verb(*args, **kwargs)
        if callable(result) and not isinstance(result, Pipe):
            return Pipe(result)
        return result

    return wrapper


def pipeable(func: Callable) -> Pipe:
    """Wrap a plain one-argument function so it supports the `|` pipe operator.

    Use as a decorator on your own pipeline steps, or inline to drop a bare lambda
    into a chain:

        @pipeable
        def clean(df):
            return df.dropna()

        data | clean | pipeable(lambda x: x * 2)

    Args:
        func (Callable): a one-argument function to make `|`-able.

    Returns:
        Pipe: a callable wrapper that also supports `data | wrapper`.
    """
    return Pipe(func)


@curry
def pop(idx):
    """Given a tuple, removes an element located at an `idx`. Useful for pruning down a
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

        return Pipe(remove)
    else:
        raise TypeError("pop requires an integer index to of the ouput to drop")


@curry
def fork(n):
    """Duplicate an input `n` number of times"""
    from copy import deepcopy

    def duplicate(data):
        copy = getattr(data, "copy", None)
        if callable(copy):
            return tuple([data.copy()] * n)
        return tuple([deepcopy(data)] * n)

    return Pipe(duplicate)


def gather(func):
    """Wraps a function that takes multiple inputs to make the output of a previous
    function with multiple outputs easier to work with. Useful after a call to `append`,
    `spread`, `across` or `mapmany` e.g.

        gather(lambda first_name, last_name: first_name + last_name)
    """

    def call(data):
        if not (isinstance(data, (list, tuple)) and len(data) > 1):
            raise TypeError(
                f"gather expects the previous step's output to be a list/tuple of length > 1 but received a {type(data)}"
            )
        return func(*data)

    return Pipe(call)


def unpack(func):
    """Alias for `gather`"""
    return gather(func)


@curry
def spread(*args):
    """Like `fork` but expects multiple functions"""

    from copy import deepcopy

    if len(args) == 1:
        if isinstance(args[0], int):

            def duplicate(data):
                copy = getattr(data, "copy", None)
                if callable(copy):
                    return tuple([data.copy()] * args[0])
                return tuple([deepcopy(data)] * args[0])

            return Pipe(duplicate)
        else:
            raise ValueError(
                "only 1 function passed to spread. Use do() instead or simply call the function directly in the pipe"
            )

    elif all(callable(a) for a in args):
        together = juxt(*args)
        return Pipe(together)
    else:
        raise TypeError("spread expected an integer or 1+ functions")


@curry
def alongwith(func):
    """Takes a function or obj and returns a new function that prepends the args to the
    function as part of the input, i.e. (input, funcval)"""

    def _alongwith(data):
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

    return Pipe(_alongwith)


@curry
def append(thing):
    """Alias for `alongwith`"""
    return alongwith(thing)
