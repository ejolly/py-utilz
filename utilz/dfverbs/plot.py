"""
plotting verbs to wrap calls to seaborn

"""

__all__ = [
    "relplot",
    "scatterplot",
    "lineplot",
    "displot",
    "histplot",
    "kdeplot",
    "ecdfplot",
    "rugplot",
    "catplot",
    "stripplot",
    "swarmplot",
    "boxplot",
    "violinplot",
    "boxenplot",
    "pointplot",
    "barplot",
    "countplot",
    "lmplot",
    "regplot",
    "residplot",
    "heatmap",
    "clustermap",
    "pairplot",
    "jointplot",
    "stripbarplot",
]

from toolz import curry
from functools import wraps
from ..plot import newax, stripbarplot as _stripbarplot
from .polars_utils import ensure_pandas_for_plotting
import seaborn as sns


def polars_plotting_wrapper(plot_func):
    """Decorator to automatically convert polars DataFrames to pandas for plotting."""
    @wraps(plot_func)
    def wrapper(data):
        # Convert polars to pandas if needed
        data = ensure_pandas_for_plotting(data)
        return plot_func(data)
    return wrapper


@curry
def jointplot(**kwargs):
    """Call to seaborn jointplot. Works with both pandas and polars DataFrames."""

    @polars_plotting_wrapper
    def plot(data):
        return sns.jointplot(data=data, ax=newax(), **kwargs)

    return plot


@curry
def pairplot(**kwargs):
    """Call to seaborn pairplot. Works with both pandas and polars DataFrames."""

    @polars_plotting_wrapper
    def plot(data):
        return sns.pairplot(data=data, **kwargs)

    return plot


@curry
def clustermap(**kwargs):
    """Call to seaborn clustermap. Works with both pandas and polars DataFrames."""

    @polars_plotting_wrapper
    def plot(data):
        return sns.clustermap(data=data, **kwargs)

    return plot


@curry
def residplot(**kwargs):
    """Call to seaborn residplot. Works with both pandas and polars DataFrames."""

    @polars_plotting_wrapper
    def plot(data):
        return sns.residplot(data=data, ax=newax(), **kwargs)

    return plot


@curry
def regplot(**kwargs):
    """Call to seaborn regplot. Works with both pandas and polars DataFrames."""

    @polars_plotting_wrapper
    def plot(data):
        return sns.regplot(data=data, ax=newax(), **kwargs)

    return plot


@curry
def lmplot(**kwargs):
    """Call to seaborn lmplot. Works with both pandas and polars DataFrames."""

    @polars_plotting_wrapper
    def plot(data):
        return sns.lmplot(data=data, **kwargs)

    return plot


@curry
def countplot(**kwargs):
    """Call to seaborn countplot. Works with both pandas and polars DataFrames."""

    @polars_plotting_wrapper
    def plot(data):
        return sns.countplot(data=data, ax=newax(), **kwargs)

    return plot


@curry
def pointplot(**kwargs):
    """Call to seaborn pointplot. Works with both pandas and polars DataFrames."""

    @polars_plotting_wrapper
    def plot(data):
        return sns.pointplot(data=data, ax=newax(), **kwargs)

    return plot


@curry
def boxenplot(**kwargs):
    """Call to seaborn boxenplot. Works with both pandas and polars DataFrames."""

    @polars_plotting_wrapper
    def plot(data):
        return sns.boxenplot(data=data, ax=newax(), **kwargs)

    return plot


@curry
def violinplot(**kwargs):
    """Call to seaborn violinplot. Works with both pandas and polars DataFrames."""

    @polars_plotting_wrapper
    def plot(data):
        return sns.violinplot(data=data, ax=newax(), **kwargs)

    return plot


@curry
def boxplot(**kwargs):
    """Call to seaborn boxplot. Works with both pandas and polars DataFrames."""

    @polars_plotting_wrapper
    def plot(data):
        return sns.boxplot(data=data, ax=newax(), **kwargs)

    return plot


@curry
def swarmplot(**kwargs):
    """Call to seaborn swarmplot. Works with both pandas and polars DataFrames."""

    @polars_plotting_wrapper
    def plot(data):
        return sns.swarmplot(data=data, ax=newax(), **kwargs)

    return plot


@curry
def stripplot(**kwargs):
    """Call to seaborn stripplot. Works with both pandas and polars DataFrames."""

    @polars_plotting_wrapper
    def plot(data):
        return sns.stripplot(data=data, ax=newax(), **kwargs)

    return plot


@curry
def rugplot(**kwargs):
    """Call to seaborn rugplot. Works with both pandas and polars DataFrames."""

    @polars_plotting_wrapper
    def plot(data):
        return sns.rugplot(data=data, ax=newax(), **kwargs)

    return plot


@curry
def ecdfplot(**kwargs):
    """Call to seaborn ecdfplot. Works with both pandas and polars DataFrames."""

    @polars_plotting_wrapper
    def plot(data):
        return sns.ecdfplot(data=data, ax=newax(), **kwargs)

    return plot


@curry
def kdeplot(**kwargs):
    """Call to seaborn kdeplot. Works with both pandas and polars DataFrames."""

    @polars_plotting_wrapper
    def plot(data):
        return sns.kdeplot(data=data, ax=newax(), **kwargs)

    return plot


@curry
def histplot(**kwargs):
    """Call to seaborn histplot. Works with both pandas and polars DataFrames."""

    @polars_plotting_wrapper
    def plot(data):
        return sns.histplot(data=data, ax=newax(), **kwargs)

    return plot


@curry
def displot(**kwargs):
    """Call to seaborn displot. Works with both pandas and polars DataFrames."""

    @polars_plotting_wrapper
    def plot(data):
        return sns.displot(data=data, **kwargs)

    return plot


@curry
def scatterplot(**kwargs):
    """Call to seaborn scatterplot. Works with both pandas and polars DataFrames."""

    @polars_plotting_wrapper
    def plot(data):
        return sns.scatterplot(data=data, ax=newax(), **kwargs)

    return plot


@curry
def relplot(**kwargs):
    """Call to seaborn relplot. Works with both pandas and polars DataFrames."""

    @polars_plotting_wrapper
    def plot(data):
        return sns.relplot(data=data, **kwargs)

    return plot


@curry
def heatmap(**kwargs):
    """Call to seaborn heatmap. Works with both pandas and polars DataFrames."""

    @polars_plotting_wrapper
    def plot(data):
        return sns.heatmap(data=data, ax=newax(), **kwargs)

    return plot


@curry
def lineplot(**kwargs):
    """Call to seaborn lineplot. Works with both pandas and polars DataFrames."""

    @polars_plotting_wrapper
    def plot(data):
        return sns.lineplot(data=data, ax=newax(), **kwargs)

    return plot


@curry
def catplot(**kwargs):
    """Call to seaborn catplot. Works with both pandas and polars DataFrames."""

    @polars_plotting_wrapper
    def plot(data):
        return sns.catplot(data=data, **kwargs)

    return plot


@curry
def barplot(**kwargs):
    """Call to seaborn barplot. Works with both pandas and polars DataFrames."""

    @polars_plotting_wrapper
    def plot(data):
        return sns.barplot(data=data, ax=newax(), **kwargs)

    return plot


@curry
def stripbarplot(**kwargs):
    """Call to combined stripplot and barplot. See utilz.plot.stripbarplot. Works with both pandas and polars DataFrames."""

    @polars_plotting_wrapper
    def plot(data):
        ax = kwargs.pop("ax", "newax")
        return _stripbarplot(data=data, ax=ax, **kwargs)

    return plot


@curry
def plot(*args, **kwargs):
    """Call a dataframe's .plot method. Works with both pandas and polars DataFrames."""

    @polars_plotting_wrapper
    def call(df):
        return df.plot(*args, **kwargs)

    return call


# Make every plot verb usable with the unix-style `|` pipe operator.
from ..pipes import _pipeify  # noqa: E402

for _name in dict.fromkeys(__all__):
    globals()[_name] = _pipeify(globals()[_name])
