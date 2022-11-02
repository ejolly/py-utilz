# Dataframe verbs and tools


The `dfverbs` module is intended to be imported as an alias and used inside `pipe` for [dplyr](https://dplyr.tidyverse.org/) like data manipulation grammar. Using the sample on the [redframes](https://github.com/maxhumber/redframes) README: 

```python
import pandas as pd
from utilz import pipe, randdf
import utilz.dfverbs as _

# Define demo df
df = pd.DataFrame({
    'bear': ['Brown bear', 'Polar bear', 'Asian black bear', 'American black bear', 'Sun bear', 'Sloth bear', 'Spectacled bear', 'Giant panda'],
    'genus': ['Ursus', 'Ursus', 'Ursus', 'Ursus', 'Helarctos', 'Melursus', 'Tremarctos', 'Ailuropoda'],
    'weight (male, lbs)': ['300-860', '880-1320', '220-440', '125-500', '60-150', '175-310', '220-340', '190-275'],
    'weight (female, lbs)': ['205-455', '330-550', '110-275', '90-300', '45-90', '120-210', '140-180', '155-220']
})

out = pipe(
    df,
    _.rename({"weight (male, lbs)": "male", "weight (female, lbs)": "female"}),
    _.to_long(columns=["male", "female"], into=("sex", "weight")),
    _.split("weight", ("min", "max"), sep="-"),
    _.to_long(columns=["min", "max"], into=("stat", "weight")),
    _.astype({"weight": float}),
    _.groupby("genus", "sex"),
    _.assign(weight="weight.mean()"),
    _.to_wide(column="sex", using="weight"),
    _.mutate(dimorphism="male / female"),  # no rounding possible
    _.mutate(dimorphism=lambda df: np.round(df.male / df.female, 2)),
)
```

!!! note

    The `dftools` module on the other handed is **not intended to be imported at all.** Instead it defines new `.methods` on pandas `DataFrame` and `DataFrameGroupBy` objects automatically, e.g. `df.select('-Col1')` is a new method that allows for R-style column selection.


## Verbs

::: utilz.dfverbs.verbs

## Stats

::: utilz.dfverbs.stats

## Plots

::: utilz.dfverbs.plot

## `utilz.dftools` 

::: utilz.dftools