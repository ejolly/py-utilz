# utilz.pipe

`utilz` proves a *pipe* operator similar to the `%>%` operator from [magrittr](https://cran.r-project.org/web/packages/magrittr/vignettes/magrittr.html) in R.

!!! note
    Doesn't currently work with numpy arrays. Other data types and pandas DataFrames are ok.

::: utilz.pipe

#### Example usage

1) First create the pipe object at the top of your code and name it whatever you want. I like 'o'

```
from utilz.pipe import Pipe
o = Pipe()

# Some data to work with
from seaborn import load_dataset
df = load_dataset('iris')
```

2) Then use it with the `>>o>>` syntax  

  - pipe to another function including lambdas (wrapped in parens)  

        df >>o>> print

        df >>o>> (lambda df: df*2)  

  - pipe to a method and call it as a string (without the '.')    

        df >>o>> 'head'  
  
  - pass `args` and `kwargs` to the method or function as a tuple. The first item in the tuple should be the method name (str) or function (callable). Subsequent items are interpreted as kwargs if they're dicts, or args if they're anything else.  
  
        df >>o>> ('mean', 1) 
        # equivalent to df.mean(1)

        df >>o>> ('mean', 1, {'numeric_only': True}) 
        # df.mean(1, numeric_only=True)

        df >>o>> (pd.melt, {'id_vars': 'species'}, {'value_vars': 'petal_length'})
        # pd.melt(df, id_vars='species', value_vars='petal_length')

  - this is the same as above since melt is both a method on DataFrames and a module function in pandas
  
        df >>o>> ('melt', {'id_vars': 'species'}, {'value_vars': 'petal_length'})
        # df.melt(id_vars='species', value_vars='petal_length')

  - You can also combine kwargs into a single dict 
  
        df >>o>> (pd.melt, {'id_vars': 'species', 'value_vars': 'petal_length'})
