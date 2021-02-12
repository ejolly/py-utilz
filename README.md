# Utilz
![Build Status](https://github.com/ejolly/utilz/workflows/Utilz/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/ejolly/utilz/badge.svg?branch=master)](https://coveralls.io/github/ejolly/utilz?branch=master)
![Python Versions](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue)
![Platforms](https://img.shields.io/badge/platform-linux%20%7C%20osx%20%7C%20win-blue)

A python package that combines several key ideas to make data analysis faster, easier, and more reliable:  

1. Declarative, concise, composable *functional-programming* style data analysis
2. Defensive data analysis through the likes of [bulwark](https://bulwark.readthedocs.io/en/latest/index.html)
3. Minimization of boilerplate code, e.g. for plotting in `matplotlib` and `seaborn`
4. Common data operations in pandas, e.g. normalizing by group
5. Common `i/o` operations like managing paths

## Examples

```python
# dplyr like verbs compatible with toolz.pipe
df = pipe(
    randdf((20, 3)),
    assign(D=list("abcde") * 4),
    rename({"A": "rt", "B": "score", "C": "speed", "D": "group"}),
    assign(rt_doubled="rt*2"),
    groupby('group'), 
    assign(
        score_centered='score - score.mean()', 
        score_norm = 'score/score.std()'
        )
    save("test"),
)
```

```python
# Special decorators to auto-cache long running function results to disk
# memoize outputs in memory, check for certain properties, and more

# Make sure all groups in 'group' have the same shape
# Auto-save a csv/h5 of the result if norm's runtime > 30s
@same_shape('group')
@disk_cache(threshold=30)
def norm(df, num='', denom=''):
    "Simulate expensive function that takes args"
    print("computing...")
    sleep(5)
    return pd.DataFrame({"norm": df[num] / df[denom]})

pipe(df, 
    groupby('group'), 
    norm(num='score', denom = 'rt') # This only runs once, then reads from disk
)

```

### Checkout the [demo notebook](https://eshinjolly.com/utilz/api/fp_data_analysis) for a complete example of what's possible