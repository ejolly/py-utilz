"""
Data containers
"""

from .maps import map


class Box(list):

    """
    Box is a flexible list-like container that allows for dot-notation access to attributes of its elements and methods. This makes it easy for example to perform a `.groupby()` operation on a list of dataframes.

    Boxes can be transparent or opaque. Transparent boxes always return the result of an operation as a list. This is useful for example when you want to call a method on each box element and immediately work with the results.

    Opaque boxes always return a new `Box` who's contents can be accessed using `.contents()` or slice notation `box[:]`. This allows for method chaining on the underlying data.

    Examples:
        >>> # Transparent box by default
        >>> box = Box([df1, df2, df3])

        >>> # Access content like a list
        >>> box[0] # returns df1
        >>> box[:] # returns [df1, df2, df3]
        >>> box.contents() # equivalent

        >>> # Access attributes or call methods just like you would on a single object
        >>> box.head(10) # returns a list each df head
        >>> box.shape # returns a list each df shape

        >>> # Opaque box facilitates method chaining but need
        >>> # `.contents()` to access results
        >>> black_box = Box([df1, df2, df3], transparent=False)
        >>> black_box.groupby('col').mean().contents()

        >>> # Apply arbitrary functions to box elements
        >>> result = box.map(lambda x: x + 1)

        >>> # Can also modify in place without returning anything
        >>> box.map(lambda x: x + 1, inplace=True)

        >>> # Map respects box transparency for method chaining
        >>> box.set_transparent(False)
        >>> result = box.map(lambda x: x + 1).head().contents()
    """

    def __init__(self, iterable, transparent=True):
        """
        Create a new box from an iterable

        Args:
            list (iterable): iterable of objects to store in the box
            transparent (bool): whether methods should return results (`True`) or a new box (`False`); Default True
        """
        super().__init__(iterable)
        self._transparent_box = transparent

    def __getattr__(self, name, *args, **kwargs):
        if hasattr(self[0], name):
            attr_or_method = getattr(self[0], name)
            if callable(attr_or_method):

                def fn(*args, **kwargs):
                    out = []
                    for elem in self:
                        result = getattr(elem, name)
                        result = result(*args, **kwargs) if callable(result) else result
                        out.append(result)
                    out = (
                        out
                        if self._transparent_box
                        else Box(out, transparent=self._transparent_box)
                    )
                    return out

                return fn
            else:
                out = [getattr(elem, name) for elem in self]
                out = (
                    out
                    if self._transparent_box
                    else Box(out, transparent=self._transparent_box)
                )
                return out

        else:
            raise AttributeError

    def __repr__(self):
        return f"Box(len={len(self)}, transparent={self._transparent_box}, type={self[0].__class__.__module__}.{self[0].__class__.__name__})"

    def map(self, fn, inplace=False):
        """
        Apply a function to each element in the box

        Args:
            fn (callable): function to apply to each element
            inplace (bool, optional): whether to modify the box in place or return a new box. Defaults to False.
            *args: positional arguments to pass to `fn`
            **kwargs: keyword arguments to pass to `fn`

        Returns:
            Box: new box with the results of applying `fn` to each element
        """

        out = map(fn, self)
        if inplace:
            self.__init__(out, transparent=self._transparent_box)
        else:
            out = (
                out
                if self._transparent_box
                else Box(out, transparent=self._transparent_box)
            )
            return out

    def contents(self):
        """
        Convert box to list

        Returns:
            list: list of elements
        """
        return list(self)

    def set_transparent(self, transparent):
        """
        Set the transparency of the box

        Args:
            transparent (bool): whether the box should be transparent or not
        """
        self._transparent_box = transparent
