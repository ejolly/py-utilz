"""
Plotting convenience functions
"""

__all__ = ["setup"]


def setup(figsize=(8, 6), subplots=(1, 1)):
    """
    Setup matplotlib subplots boilerplate

    Args:
        figsize (tuple, optional): Figure size. Defaults to (8, 6).
        subplots (tuple, optional): subplot grid size. Defaults to (1, 1).

    Returns:
        tuple ((Figure, Axes)): matplotlib figure handle and axes
    """
    if "plt" not in dir():
        import matplotlib.pyplot as plt
    f, ax = plt.subplots(*subplots, figsize=figsize)
    return f, ax
