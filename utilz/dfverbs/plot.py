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
from ..plot import newax, stripbarplot as _stripbarplot
import seaborn as sns


@curry
def jointplot(**kwargs):
    """Call to seaborn jointplot"""

    def plot(data):
        return sns.jointplot(data=data, ax=newax(), **kwargs)

    return plot


@curry
def pairplot(**kwargs):
    """Call to seaborn pairplot"""

    def plot(data):
        return sns.pairplot(data=data, **kwargs)

    return plot


@curry
def clustermap(**kwargs):
    """Call to seaborn clustermap"""

    def plot(data):
        return sns.clustermap(data=data, **kwargs)

    return plot


@curry
def residplot(**kwargs):
    """Call to seaborn residplot"""

    def plot(data):
        return sns.residplot(data=data, ax=newax(), **kwargs)

    return plot


@curry
def regplot(**kwargs):
    """Call to seaborn regplot"""

    def plot(data):
        return sns.regplot(data=data, ax=newax(), **kwargs)

    return plot


@curry
def lmplot(**kwargs):
    """Call to seaborn lmplot"""

    def plot(data):
        return sns.lmplot(data=data, **kwargs)

    return plot


@curry
def countplot(**kwargs):
    """Call to seaborn countplot"""

    def plot(data):
        return sns.countplot(data=data, ax=newax(), **kwargs)

    return plot


@curry
def pointplot(**kwargs):
    """Call to seaborn pointplot"""

    def plot(data):
        return sns.pointplot(data=data, ax=newax(), **kwargs)

    return plot


@curry
def boxenplot(**kwargs):
    """Call to seaborn boxenplot"""

    def plot(data):
        return sns.boxenplot(data=data, ax=newax(), **kwargs)

    return plot


@curry
def violinplot(**kwargs):
    """Call to seaborn violinplot"""

    def plot(data):
        return sns.violinplot(data=data, ax=newax(), **kwargs)

    return plot


@curry
def boxplot(**kwargs):
    """Call to seaborn boxplot"""

    def plot(data):
        return sns.boxplot(data=data, ax=newax(), **kwargs)

    return plot


@curry
def swarmplot(**kwargs):
    """Call to seaborn swarmplot"""

    def plot(data):
        return sns.swarmplot(data=data, ax=newax(), **kwargs)

    return plot


@curry
def stripplot(**kwargs):
    """Call to seaborn stripplot"""

    def plot(data):
        return sns.stripplot(data=data, ax=newax(), **kwargs)

    return plot


@curry
def rugplot(**kwargs):
    """Call to seaborn rugplot"""

    def plot(data):
        return sns.rugplot(data=data, ax=newax(), **kwargs)

    return plot


@curry
def ecdfplot(**kwargs):
    """Call to seaborn ecdfplot"""

    def plot(data):
        return sns.ecdfplot(data=data, ax=newax(), **kwargs)

    return plot


@curry
def kdeplot(**kwargs):
    """Call to seaborn kdeplot"""

    def plot(data):
        return sns.kdeplot(data=data, ax=newax(), **kwargs)

    return plot


@curry
def histplot(**kwargs):
    """Call to seaborn histplot"""

    def plot(data):
        return sns.histplot(data=data, ax=newax(), **kwargs)

    return plot


@curry
def displot(**kwargs):
    """Call to seaborn displot"""

    def plot(data):
        return sns.displot(data=data, **kwargs)

    return plot


@curry
def scatterplot(**kwargs):
    """Call to seaborn scatterplot"""

    def plot(data):
        return sns.scatterplot(data=data, ax=newax(), **kwargs)

    return plot


@curry
def relplot(**kwargs):
    """Call to seaborn relplot"""

    def plot(data):
        return sns.relplot(data=data, **kwargs)

    return plot


@curry
def heatmap(**kwargs):
    """Call to seaborn heatmap"""

    def plot(data):
        return sns.heatmap(data=data, ax=newax(), **kwargs)

    return plot


@curry
def lineplot(**kwargs):
    """Call to seaborn lineplot"""

    def plot(data):
        return sns.lineplot(data=data, ax=newax(), **kwargs)

    return plot


@curry
def catplot(**kwargs):
    """Call to seaborn catplot"""

    def plot(data):
        return sns.catplot(data=data, **kwargs)

    return plot


@curry
def barplot(**kwargs):
    """Call to seaborn barplot"""

    def plot(data):
        return sns.barplot(data=data, ax=newax(), **kwargs)

    return plot


@curry
def stripbarplot(**kwargs):
    """Call to combined stripplot and barplot. See utilz.plot.stripbarplot"""

    def plot(data):
        ax = kwargs.pop("ax", "newax")
        return _stripbarplot(data=data, ax=ax, **kwargs)

    return plot


@curry
def plot(*args, **kwargs):
    """Call a dataframe's .plot method"""

    def call(df):
        return df.plot(*args, **kwargs)

    return call
