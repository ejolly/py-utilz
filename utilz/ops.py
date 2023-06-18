"""
Functional tools intended to be used with one or more functions and a single object. 

## Overview

| function (s)   | description  | 
|---|---|
| `do`  | apply a function or method to an object |
| `many`  | applies **multiple functions indepedently** to a **single** input and returns an iterable; like an "inverse `map`" |
| `compose`  | combine **multiple functions** into **one function** sequence (mini-pipe) |
| `iffy`  | apply a function if a predicate function is true otherwise noop |

---
"""
__all__ = [
    "many",
    "do",
    "iffy",
    "compose",
    "curry",
]

from typing import Union, Any
from collections.abc import Callable
from toolz import curry
from toolz.curried import compose_left


def many(funcs, data):
    """
    Takes an iterable of funcs and applies them each separately to data. Operates like the inverse of map(). Whereas map takes applies 1 func to multiple elements, many applies multiple funcs independently to 1 element. Returns a tuple the same length as funcs, containing the output of each function


    Args:
        funcs (iterable): iterable of functions to apply
        data (any): data to aplly functions to


    Returns:
        list: separate function evaluations
    """

    if isinstance(data, (list, tuple)):
        raise TypeError(
            f"Expected a single input but receive {len(data)}. Use mapmany() to operate on an iterable"
        )
    if not isinstance(funcs, (list, tuple)) or len(funcs) <= 1:
        raise ValueError(
            f"many applies *multiple* function calls separately but only received a single function. Use do() to apply a single function or method."
        )

    return [do(f, data) for f in funcs]


@curry
def do(func, data, *args, **kwargs):
    """
    Apply a single function to data or call a method on data, while passing optional
    kwargs to that functinon or method

    Args:
        func (callable/str): callable or str method name
        data (any): object to apply function to
        kwargs (any): additional arguments to function or method

    Returns:
        any: function or method evaluation
    """
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

    Returns:
        any: function evaluation or original data
    """

    if not callable(predicate_func):
        raise TypeError("iffy requires a function that returns a boolean value")

    if predicate_func(data):
        if callable(if_true):
            return if_true(data)
        else:
            return if_true
    else:
        return data


@curry
def compose(*funcs):
    """
    Compose multiple functions into a single function

    Args:
        funcs (any): any number of functions listed as separate args
    Returns:
        callable: new composed function
    """
    return compose_left(*funcs)
