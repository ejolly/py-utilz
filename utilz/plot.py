"""
Plotting convenience functions
"""

__all__ = ["stripbarplot"]

import seaborn as sns


def stripbarplot(
    data,
    pointcolor="black",
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
):
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
    ax = sns.barplot(*args, **kwargs, data=data)
    ax = sns.stripplot(*args, **kwargs, color=pointcolor, data=data, ax=ax)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xticklabels:
        ax.set_xticklabels(xticklabels)
    if yticklabels:
        ax.set_yticklabels(yticklabels)
    if xticks:
        ax.set_xticks(xticks)
    if yticks:
        ax.set_yticks(yticks)
    if xlim:
        ax.set(xlim=xlim)
    if ylim:
        ax.set(ylim=ylim)
    return ax
