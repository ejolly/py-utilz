"""
dplyr like *verbs* for working with pandas dataframes. Designed to be piped together using the `pipe` function from the `toolz` package. Provides a broad use-case alternative to plydata or larger packages while use mostly native pandas methods under-the-hood.

*Note*: the current version unfortunately does borrow the select, and define functions from plydata

# Workhorses

- **rows**: subset rows based on some conditions (str), ranges (tuples), or indices (list)
- **cols**: subset cols based on some conditions (str), ranges (tuples), or indices (list); support "-col" inversion -> *from plydata*
- **summarize**: create a new column(s) thats the result of a operation that returns a *scalar* value and assigns it back to df. Accepts dfs or grouped dfs
- **assign**: create a new column(s) thats the result of a operation that returns a *series* value and assigns it back to df. Accepts dfs or grouped dfs

*I think summarize -> aggregate and assign -> mutate in dplyr land*  

# Secondary
- **apply**: to apply artbitrary functions on a df or grouped df (just a wrapper around df.apply)

# Example Verb-based analysis of a dataframe

```python
from toolz import pipe
from utilz import rows, cols

# Basic slicing/subsetting

pipe(x, 
    rows("group == 'c' or group == 'b'"), 
    cols("rt", "speed")
    )

pipe(x, 
    rows([1,9,14]), 
    cols((3,5))
    )

pipe(x, 
    rows((1,11,2))
    )

# Perform scalar operation that always return back a *smaller* dataframe or series than the original
# Non-grouped inputs produce series results, while grouped inputs produce dataframe results

pipe(x, 
    rows("group == 'c' or group == 'b'"), 
    summarize(rt='mean',speed='mean')
    )

pipe(x, 
    groupby('group'), 
    summarize(score = 'mean', rt = 'std')
    )

# Perform series operations that always return back a dataframe thats the *same* size as the original
# To preserve input size, use assign instead. It will "broadcast" a smaller set of values over the length of the entire input dataframe, while respecting groups if the input is grouped
pipe(x, assign(score_centered='score.mean()'))
# In this case the return values are computed and broadcasted within group rather than across the entire frame

pipe(x, 
    groupby('group'), 
    assign(
        score_centered='score - score.mean()', 
        score_norm = 'score/score.std()'
        )
        
    )

# e.g. the mean in sub-group

pipe(x, 
    groupby('group'), 
    assign(speed_per_group='speed.mean()')
    )
```

"""

__all__ = ["groupby", "rows", "cols", "rename", "save", "summarize", "assign", "apply"]

from cytoolz import curry
import numpy as np
import pandas as pd


# We curry these because intend them to be used with pipe which implicitly passes an argument in
# df is second for functions that don't have keyword args because thats how each function will receive it from pipe
@curry
def groupby(cols, df):
    """Call a dataframe's `.groupby` method"""
    return df.groupby(cols)


@curry
def rows(query, df):
    """Select rows using a query (str), slice (tuple), or indices (list) """
    if isinstance(query, str):
        return df.query(query).reset_index(drop=True)
    elif isinstance(query, (list, np.ndarray)):
        return df.iloc[query, :]
    elif isinstance(query, tuple):
        return df.iloc[slice(*query), :]


@curry
def cols(query, df):
    """Select columns using a query (str), slice (tuple), or indices (list) """
    if isinstance(query, str):
        from plydata import select

        return select(df, query)
    elif isinstance(query, (list, np.ndarray)):
        return df.iloc[:, query]
    elif isinstance(query, tuple):
        return df.iloc[:, slice(*query)]


@curry
def rename(cols, df):
    """Call a dataframe's `.rename(columns={})` method"""
    return df.rename(columns=cols)


@curry
def save(path, df):
    """Call a dataframe's `.to_csv(index=False)` method"""
    df.to_csv(f"{path}.csv", index=False)
    return df


@curry
def summarize(df, **stats):
    """Call a dataframe or groupby object's `.agg` method"""
    return df.agg(stats)


@curry
def assign(dfg, *args, **kwargs):
    """
    Creates a new column(s) in df based on a function of existing columns in df.
    Uses define (i.e. "mutate") in plydata unless the input is a grouped dataframe
    in which case it falls back to pandas methods because plydata can only handle
    grouped inputs resulting from its own (slow) group_by function
    """

    if isinstance(dfg, pd.core.groupby.generic.DataFrameGroupBy):
        prev = dfg.filter(lambda _: True).reset_index()
        for _, (k, v) in enumerate(kwargs.items()):
            res = dfg.apply(lambda group: group.eval(v)).reset_index()
            group_col = res.columns[0]
            if isinstance(res, pd.DataFrame) and "level_1" in res.columns:
                res.columns = [res.columns[0], "index"] + [k]
                prev = prev.merge(res.drop(columns=group_col), on="index")
            else:
                res.columns = [res.columns[0]] + [k]
                prev = prev.merge(res, on=group_col)
        prev = prev.sort_values(by="index").drop(columns="index")
        return prev
    else:
        from plydata import define

        return define(dfg, *args, **kwargs)


@curry
def apply(func, df):
    """Call a dataframe or groupby object's `.apply` method"""
    return df.apply(func)
