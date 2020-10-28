"""
Boilerplate plotting code
"""


def init_termplot():
    import matplotlib

    matplotlib.use("module://imgcat")


def s():
    import matplotlib.pyplot as plt

    """
    Show a plot in the terminal and immediately close the figure handle to save memory.
    Usage:

    Example 1:

    plt.plot([1,2,3])
    s()
    
    Example 2:

    sns.scatterplot('x','y',data=df)
    s()
    """
    plt.show()
    plt.close()


def p(obj, *args, **kwargs):
    """
    Show a plot in the terminal using an object's own .plot method.

    Example 1:

    df = pd.DataFrame({'A': [1,2,3], 'B': [4,5,6]})
    p(df)

    Example 2:

    f, ax = plt.subplots(1, 1, figsize=(4, 3))
    out = plot(ax, [1, 2, 3])
    """
    out = obj.plot(*args, **kwargs)
    s()
    return out


def setup(figsize=(8, 6), subplots=(1, 1)):
    import matplotlib.pyplt as plt

    f, ax = plt.subplots(*subplots, figsize=figsize)
    return f, ax
