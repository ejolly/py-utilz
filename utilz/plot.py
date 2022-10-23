"""
Plotting convenience functions
"""

__all__ = ["mpinit", "stripbarplot", "savefig", "tweak", "newax"]

import seaborn as sns
from pathlib import Path
from matplotlib.figure import Figure, Axes
import matplotlib.pyplot as plt
import numpy as np
from toolz import curry
from typing import Union


def mpinit(figsize: tuple = (8, 6), subplots: tuple = (1, 1)):
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


@curry
def stripbarplot(
    data,
    pointcolor="black",
    remove_duplicate_legend=True,
    xlabel=None,
    ylabel=None,
    xticklabels=None,
    yticklabels=None,
    xticks=None,
    yticks=None,
    xlim=None,
    ylim=None,
    *args,
    **kwargs,
) -> Axes:
    """
    Combines a call to `sns.barplot` + `sns.stripplot`. Optionally set some axis level attributes during plot creation. Leaving these attributes None will return the default labels that seaborn sets.

    Args:
        data (DataFrame): input data
        pointcolor (str, optional): color of stripplot points. Defaults to "black".
        xlabel ([type], optional): x-axis label. Defaults to seaborn's default.
        ylabel ([type], optional): Defaults to seaborn's default.
        xticklabels ([type], optional):  Defaults to seaborn's default.
        yticklabels ([type], optional):  Defaults to seaborn's default.
        xticks ([type], optional):  Defaults to seaborn's default.
        yticks ([type], optional):  Defaults to seaborn's default.
        xlim ([type], optional):  Defaults to seaborn's default.
        ylim ([type], optional):  Defaults to seaborn's default.

    Returns:
        Axis: plot axis handle
    """
    ax = kwargs.pop("ax", None)
    estimator = kwargs.pop("estimator", np.mean)
    ncol = kwargs.pop("ncol", None)
    loc = kwargs.pop("loc", None)
    legend = kwargs.pop("legend", None)
    alpha = kwargs.pop("alpha", 1)

    ax = sns.barplot(*args, **kwargs, data=data, ax=ax, estimator=estimator)
    ax = sns.stripplot(*args, **kwargs, color=pointcolor, data=data, ax=ax, alpha=alpha)

    if legend is False:
        _ = ax.get_legend().remove()

    elif remove_duplicate_legend:
        handles, labels = ax.get_legend_handles_labels()
        half = int(len(handles) / 2)
        if ncol is None:
            if loc is None:
                legend = ax.legend(handles[half:], labels[half:])
            else:
                legend = ax.legend(handles[half:], labels[half:], loc=loc)

        elif ncol is not None:
            if loc is None:
                legend = ax.legend(handles[half:], labels[half:], ncol=ncol)
            else:
                legend = ax.legend(handles[half:], labels[half:], ncol=ncol, loc=loc)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if xlim is not None:
        ax.set(xlim=xlim)
    if ylim is not None:
        ax.set(ylim=ylim)
    return ax


@curry
def savefig(
    f: Figure,
    name: str,
    path: Path = None,
    raster: bool = True,
    vector: bool = True,
    use_subdirs: bool = True,
    raster_extension: str = "jpg",
    bbox_inches: str = "tight",
    overwrite: bool = True,
    **kwargs,
) -> None:
    """
    Quick figure saving function. Saves raster (jpg) and vector (pdf) by default. Can
    also optionally prevent file-overwriting

    Args:
        f (Figure): matplotlib figure handle
        path (Path, optional): directory to save figure as a Path object. Defaults to
        None which will save in cwd
        name (str): filename without extension
        raster (bool, optional): whether to save raster file. Defaults to True.
        vector (bool, optional): whether to save vector file. Defaults to True.
        use_subdirs (bool, optional): whether to split saving of raster and vector files
        into subdirectories called 'raster' and 'vector'. Defaults to True.
        raster_extension (str, optional): raster file type. Defaults to "jpg".
        bbox_inches (str, optional): see bbox_inches in plt.savefig. Defaults to "tight".
        overwrite (bool, optional): whether to overwrite any existing files. Defaults to True.

    """
    if isinstance(f, Axes):
        f = f.get_figure()
    if path is not None:
        if not isinstance(path, Path):
            raise TypeError("path must be a `pathlib.Path` object")
    else:
        path = Path.cwd()
    if use_subdirs:
        raster_path = path / "raster" / f"{name}.{raster_extension}"
        vector_path = path / "vector" / f"{name}.pdf"
    else:
        raster_path = path / f"{name}.{raster_extension}"
        vector_path = path / f"{name}.pdf"
    if not raster_path.parent.exists():
        raster_path.parent.mkdir()
    if not vector_path.parent.exists():
        vector_path.parent.mkdir()
    if vector:
        if (vector_path.exists() and overwrite) or (not vector_path.exists()):
            f.savefig(vector_path, bbox_inches=bbox_inches, **kwargs)
    if raster:
        if (raster_path.exists() and overwrite) or (not raster_path.exists()):
            f.savefig(raster_path, bbox_inches=bbox_inches, **kwargs)
    return f


@curry
def tweak(plot: Union[Figure, Axes], **kwargs) -> Union[Figure, Axes]:
    """
    swiss-army knife to quickly change most aesthetics on a plot, e.g. tick labels,
    fontsize, etc, in a unified function call
    """
    if isinstance(plot, Axes):
        plot.set(**kwargs)
        return plot


def newax(*args, **kwargs):
    """Short hand for a new axis on a new figure. Usueful for calling multiple plotting
    routines in a pipe() but wanting separate figures."""
    return plt.subplots()[1]
