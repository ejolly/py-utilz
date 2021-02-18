"""
Custom datastructures often sub-classed from existing types
"""

from .ops import pmap

__all__ = ["List"]


class List(list):
    """
    A modified list sub-class that adds several standard functions as methods instead. Handy incase you prefer using a more '.' syntax style. For example, instead of `map(myfunc, mylist)`, you can instead do `mylist.map(myfunc)`. All operations return a copy and never modify the original list, which make method chaining possible, e.g. `mylist.map(somefunc).filter(anotherfunc)`
    """

    def __init__(self, iterable=[]):
        super().__init__(iterable)

    def map(self, func):
        """Apply a function to each list element."""
        return List(map(func, self))

    def pmap(self, func, *args, **kwargs):
        """Apply a function to each list element with parallelization."""

        raise NotImplementedError
        kwargs["loop_idx"] = False
        return pmap(func, self, *args, **kwargs)

    def filter(self, func):
        """Filter the elements of a list based on a function."""
        return List(filter(func, self))

    def len(self):
        """Return the length of the list"""
        return len(self)
