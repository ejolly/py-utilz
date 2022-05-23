# Experimental API

!!! warning
    Currently functions listed here require optional dependencies and/or have partial or no tests so should be used with caution. 

!!! note
    Currently `select` and `define` are borrowed from [`plydata`](https://plydata.readthedocs.io/en/stable/)

## Data analysis verbs

See this [functional programming-style data analysis notebook](fp_data_analysis.ipynb) for example usage

`groupby`: Call a dataframe's `.groupby` method

`rows`: Select rows using a `.query` (str), slicerange (start,stop,step), or indices (list)

`cols`: Select columns using a `.query` (str), slicerange (start,stop,step), or indices (list). Uses `plydata.select`

`rename`: Call a dataframe's `.rename(columns={})` method

`save`: Call a dataframe's `.to_csv(index=False)` method

`summarize`: Call a dataframe or groupby object's `.agg` method

`assign`: Creates a new column(s) in a DataFrame based on a function of existing columns in the DataFrame. Uses `plydata.define/mutate` unless the input is a grouped DataFrame in which case it falls back to pandas methods because `plydata` can only handle grouped inputs resulting from its own (slow) `group_by` function

`apply`: Call a dataframe or groupby object's `.apply` method
