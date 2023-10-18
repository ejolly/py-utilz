"""
    Tools for working with generators. Also see the `toolz` library!
"""


def make_gen(iterable):
    """Turn any iterable into a generator"""
    return (e for e in iterable)


def combine_gens(*iterables):
    """
    Combine multiple generators into a single generator based on every
    unique combination of their elements. This is equivalent to a series of
    nested for loops just like itertools.product(). But unlike itertools.product
    doesn't exhaust each generator before combining them.

    Examples:
        >>> # This
        >>> for aa in a:
        >>>   for bb in b:
        >>>     for cc in c:
        >>>       func(aa, bb, cc)

        >>> # Becomes this
        >>> for aa, bb, cc in combine_gens(a, b, c):
        >>>   func(aa, bb, cc)

    Yields:
        generator
    """
    if not iterables:
        # Base case: If no generators are provided, yield an empty tuple.
        yield ()
    else:
        # Recursive case: Combine the first generator with combinations from the rest.
        first_gen, *remaining_gens = iterables
        for item in first_gen:
            for sub_combination in combine_gens(*remaining_gens):
                yield (item,) + sub_combination
