# py-utilz

![Build Status](https://github.com/ejolly/py-utilz/workflows/Utilz/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/ejolly/utilz/badge.svg?branch=master)](https://coveralls.io/github/ejolly/utilz?branch=master)
![Python Versions](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue)
![Platforms](https://img.shields.io/badge/platform-linux%20%7C%20osx%20%7C%20win-blue)

Convenient helper functions, decorators, and data analysis tools to make life easier with minimal dependencies:

`pip install py-utilz`

[dplyr](https://dplyr.tidyverse.org/) like data grammar:

```python
from utilz import pipe
import utilz.dfverbs as _

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

More convenient (parallel) looping:

```python
from utilz import map

# Combine function results into a list, array, or dataframe
map(myfunc, myiterable)

# Syntactic sugar for joblib.Parallel
map(myfunc, myiterable, n_jobs=4)
```

Useful decorators for data analysis:

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

Checkout the [overview](intro.ipynb) page for more!
