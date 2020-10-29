"""
Plotting module dedicated to working with plots in an interactive terminal (not jupyter notebook!)
"""
__all__ = ["init_termplot", "s", "p"]


def init_termplot():
    """
    Initilize terminal based plotting. **Import and run this before any other python plotting module!** e.g. before matplotlib.

    Requires the `imgcat` command line program available in Iterm.
    """
    import matplotlib

    matplotlib.use("module://imgcat")


def s():
    """
    Show a plot in the terminal and immediately close the figure handle to save memory.
    This **has to be called** immediately after any normal python plotting function in order to render a plot in the terminal.

    The exception is if you're using `utilz.termplot.p()`, which will automatically call this function

    Examples:

        >>> plt.plot([1,2,3])
        >>> s()

        >>> sns.scatterplot('x','y',data=df)
        >>> s()
    """

    if "plt" not in dir():
        import matplotlib.pyplot as plt

    if len(plt.get_fignums()):
        plt.show()
        plt.close()
    else:
        raise ValueError(
            "No matplotlib figures found. Are you sure you plotted something?"
        )


def p(obj, *args, **kwargs):
    """
    Show a plot in the terminal using an object's own .plot method.

    No need to call `utilz.termplot.s()` if using this function.

    Args:
        obj (Any): a python object that has a `.plot()` method
        args (Any): arguments to the object's `.plot()` method
        kwargs (Any): keyword arguments to the object's `.plot()` method

    Examples:

        >>> df = pd.DataFrame({'A': [1,2,3], 'B': [4,5,6]})
        >>> p(df)

        >>> f, ax = plt.subplots(1, 1, figsize=(4, 3))
        >>> out = plot(ax, [1, 2, 3])
    """
    plot = getattr(obj, "plot", None)
    if callable(plot):
        out = obj.plot(*args, **kwargs)
        s()
        return out
    else:
        raise TypeError(f"Object of type {type(obj)} has not .plot() method!")
