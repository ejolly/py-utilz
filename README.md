# Utilz
![Build Status](https://github.com/ejolly/utilz/workflows/Utilz/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/ejolly/utilz/badge.svg?branch=master)](https://coveralls.io/github/ejolly/utilz?branch=master)
![Python Versions](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue)
![Platforms](https://img.shields.io/badge/platform-linux%20%7C%20osx%20%7C%20win-blue)

Convenient helper functions, decorators, and data analysis tools to make life easier with minimal dependencies:

`pip install py-utilz`

```python
from utilz import mapcat

# Combine function results into a list, array, or dataframe
mapcat(myfunc, myiterable) 

# Syntactic sugar for joblib.Parallel
mapcat(myfunc, myiterable, n_jobs=4)
```

```python
from utilz import log, maybe

# Print the shape of args and outputs before and after execute
@log
def myfunc(args):
    return out

# Only run myfunc if results.csv doesn't eist
@maybe('results.csv')
def myfunc(args):
    return out
```


## Development

1. `pip install -r requirements-dev.txt`
2. `pip install -r requirements-optional.txt`

Run all tests: `pytest`  
Live render docs: `mkdocs serve`  