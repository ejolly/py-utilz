"""
import pandas as pd
from pathlib import path
from cytoolz import memoize, curry, pipe
from enguarde import unique, same_size, fixed_size

# Functions
# We use memoize to when doing interactive data analysis repeatedly calling the same func with the same args just loads a cached result instead of re-running
@memoize
def load_data(path):
    full_path = path / ‘data’ / ‘subjects.csv’
    return pd.read_csv(full_path)

@memoize
@curry
@unique(group)
@same_size(group)
def zscore_by_group(df, group, value):
    return df.groupy(group).transform(lambda g: (g[value] - g[value].mean()) / g[value].std())

@memoize
@curry
@fixed_size(df.group.nunique(), features)
def lm_by_group(df, features, obs):
    return df.groupby(group).apply(lambda g: np.linalg.linregress(g[features], g[obs])


# Run the pipeline
result = pipe(‘my/project/folder’,
load_data,
zscore_by_group(‘category’, ’reaction_time’),
lm_by_group([‘color’,’size’,’shape’], ’reaction_time’)

# Or with custom pipe
result = ('my/project/folder'
            >>o>> load_data
            >>o>> zscore_by_group('category','reaction_time)
            >>o>> lm_by_group([‘color’,’size’,’shape’], ’reaction_time’)
            )
"""