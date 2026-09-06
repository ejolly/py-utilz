# py-utilz

![Build Status](https://github.com/ejolly/py-utilz/workflows/Utilz/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/ejolly/utilz/badge.svg?branch=master)](https://coveralls.io/github/ejolly/utilz?branch=master)
![Python Versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![Platforms](https://img.shields.io/badge/platform-linux%20%7C%20osx%20%7C%20win-blue)

Convenient helper functions, decorators, and data analysis tools to make life easier with minimal dependencies:

`pip install py-utilz`

## 🎉 Now with Polars Support!

The [dplyr](https://dplyr.tidyverse.org/)-like data grammar now works with both **pandas** and **polars** DataFrames:

```python
from utilz import pipe
import utilz.dfverbs as _

# Works with pandas
import pandas as pd
df = pd.read_csv("data.csv")

# Also works with polars!
import polars as pl
df = pl.read_csv("data.csv")

# Same syntax for both libraries
out = pipe(
    df,
    _.rename({"weight (male, lbs)": "male", "weight (female, lbs)": "female"}),
    _.pivot_longer(columns=["male", "female"], into=("sex", "weight")),
    _.split("weight", ("min", "max"), sep="-"),
    _.pivot_longer(columns=["min", "max"], into=("stat", "weight")),
    _.astype({"weight": float}),
    _.groupby("genus", "sex"),
    _.mutate(weight="weight.mean()"),
    _.pivot_wider(column="sex", using="weight"),
    _.mutate(dimorphism="male / female")
)
```

### Key Features with Polars

- **Automatic DataFrame detection**: All verbs automatically detect whether you're using pandas or polars
- **Consistent API**: Same function names and syntax work with both libraries
- **Performance benefits**: Leverage polars' speed while keeping familiar dplyr-like syntax
- **Plotting support**: Automatic conversion to pandas for seaborn compatibility

```python
# Polars example with plotting
import polars as pl
import utilz.dfverbs as _

df_polars = pl.DataFrame({
    'x': [1, 2, 3, 4, 5],
    'y': [2, 4, 6, 8, 10],
    'group': ['A', 'A', 'B', 'B', 'A']
})

# All dfverbs work seamlessly
result = pipe(
    df_polars,
    _.groupby('group'),
    _.summarize(mean_y='y.mean()', count='y.count()'),
    _.barplot(x='group', y='mean_y')  # Automatic conversion for plotting!
)
```

```python
from utilz import map

# Combine function results into a list, array, or dataframe
map(myfunc, myiterable)

# Syntactic sugar for joblib.Parallel
map(myfunc, myiterable, n_jobs=4)
```

```python
from utilz import log, maybe

# Print the shape of args and outputs before and after execute
@log
def myfunc(args):
    return out

# Only run myfunc if results.csv doesn't eist
@maybe
def myfunc(args, out_file=None):
    return out

myfunc(args, out_file='results.csv')
```

## Development

1. Install [uv](https://docs.astral.sh/uv/): `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Setup virtual environment `uv sync --dev`
3. Run all tests: `uv run pytest`
4. Live render docs: `uv run mkdocs serve`

### Additional uv/virtual environment commands

- Activate environment in current shell: `source .venv/bin/activate`
- Add additional packages: `uv add package_name`
- Add dev packages: `uv add --dev package_name`
- Remove packages: `uv remove package_name`
- Build local package: `uv build`
- Run any command in venv: `uv run <command>`
