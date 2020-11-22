# Efficient functional-programming-style data analysis

```python
from utilz import clear_cache, disk_cache, load
from toolz import memoize, curry, pipe
```

First clear any existing cache, i.e. the `.utilz_cache` folder.  

```python
clear_cache()
```

## Memoization

**Memoize** a function to save its last input in RAM and recall it when called with the same inputs rather than re-executing a potentially long running function

```python
@memoize
def load(path):
    "Simulate loading a slow large dataset"
    print("loading from disk")
    sleep(5)
    return pd.read_csv(path)
```

## Partial function application (currying)

**Curry** so a function operates normally when it receives all its required arguments, but turns into a *partial* function when it gets fewer that all its required arguments. This partial function behaves just like the original except with a subset of its arguments fixed. This also allows any functions to work with `toolz.pipe()`, while still being able to manipulate a function's arguments. That's because `pipe()` implicitly passes the output of the last function as the input of the next. 

```python

@curry
def calc_mean(df, normalize=False):
    ...

# With no kwargs you sometimes have to write the args backwards
@curry
def calc_mean(norm_value, df):
    ...

# Now this works, otherwise it would complain about the wrong number of arguments
pipe(df,
    calc_mean(normalize=True)
)

```

## Cacheing outputs to disk

**Cache** so the result of a function is stored to disk in file made unique by the args and kwargs to the function. Use `utilz.disk_cache` to decorate a function so it caches, which works just like `toolz.memoize` but stores the result to a file and loads the file when called with the same inputs. Better that `memoize` for larger memory hungry inputs and outputs, and necessary if input or outputs cannot be pickled (e.g. dataframes, arrays, deep objects, etc). Setting the threshold to something like 1 or 0 essentially always caches the result. 

```python
@curry
@disk_cache(threshold=1)
def norm(df, num='', denom=''):
    "Simulate expensive normalization function that takes args"
    print("computing...")
    sleep(5)
    return pd.DataFrame({"norm": df[num] / df[denom]})
```

# Together

Because `load` is memoized by default, only the first run of this pipeline actually loads the data and incurs the i/o cost. 

Because `norm` is decorated by `disk_cache`, only the first run with the same prior pipeline steps incurs a compute cost.

```python
summary = pipe(
    "test.csv",
    load,
    groupby("D"),
    assign(a_centered="A - B.mean()", mean_C="mean(C)", std_C="std(C)"), 
    norm(num='A',denom='B')
)

# Using the joblib.Memory module also does disk-cacheing, but 
# for some reason doesn't seem to work with currying
```

This setup is nice because it allows for *both* interactive data analysis as well as reproducible scripts. Simply start writing the pipeline steps, and comment out ones you want to skip or debug. In another notebook cell edit the source code of a function in the pipeline and incrementally add to its body, while rerunning the pipeline to see results as you build up your functions. 

For functions that take a while to run, try wrapping them in `memoize` or `disk_cache`. Memoize is nice for loading csv/text files (so you don't need to re-read them from disk each re-run of the pipeline). Cacheing is nice for expensive operations or operations on complex datastructures like arrays and dataframes. Plus, `utilz` saves them in standard robust file types (`.csv`. or .`h5`) files so you're also getting incremental backups of your work. No more need to rely on saved "state" in a Juptyer notebook.
