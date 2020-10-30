import pandas as pd
from utilz.guards import log_df, disk_cache
from time import sleep
from pathlib import Path
import datetime as dt

# Load iris dataset from Seaborn's data repo
df = pd.read_csv(
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
)


def test_log_df(capsys):
    @log_df
    def group_mean(df, grp_col, val_col):
        return df.groupby(grp_col)[val_col].mean().reset_index()

    _ = group_mean(df, "species", "petal_length")
    captured = capsys.readouterr()
    assert "Func group_mean df shape=(3, 2)" in captured.out


# TODO: Update test to support new caching
def test_disk_cache(tmp_path):
    # Dataframe
    @disk_cache(threshold=5, verbose=True)
    def my_long_func(df, arg, my_kwarg1="hi", my_kwarg2=False):
        sleep(6)
        return df

    key = "my_long_func_([2], (('my_kwarg1', 'hi'), ('my_kwarg2', False)))"
    fpath = Path(f"{key}.csv")
    # Remove files from failed tests
    if fpath.exists():
        fpath.unlink()
    tic = dt.datetime.now()
    _ = my_long_func(df, 2, my_kwarg1="hi", my_kwarg2=False)
    dur1 = dt.datetime.now() - tic
    dur1 = dur1.seconds
    assert Path(fpath).exists()

    tic = dt.datetime.now()
    my_output = my_long_func(df, 2, my_kwarg1="hi", my_kwarg2=False)
    dur2 = dt.datetime.now() - tic
    dur2 = dur2.seconds
    assert dur2 < dur1
    assert my_output.equals(df)
    # Remove files from successful tests
    fpath.unlink()

    # Other dtypes
    @disk_cache(threshold=5, verbose=True)
    def my_long_func(x, arg):
        sleep(6)
        return x

    key = "my_long_func_([True], ())"
    fpath = Path(f"{key}.h5")

    # Remove files from failed tests
    if fpath.exists():
        fpath.unlink()
    tic = dt.datetime.now()
    my_input = [1, 2, 3]
    _ = my_long_func(my_input, True)
    dur1 = dt.datetime.now() - tic
    dur1 = dur1.seconds
    assert Path(fpath).exists()

    tic = dt.datetime.now()
    my_output = my_long_func(my_input, True)
    dur2 = dt.datetime.now() - tic
    dur2 = dur2.seconds
    assert dur2 < dur1
    assert my_output == my_input
    # Remove files from successful tests
    fpath.unlink()
