import pandas as pd
import numpy as np
from utilz.guards import log_df, disk_cache, _hashobj, maybe
from time import sleep
from pathlib import Path
import datetime as dt
from shutil import rmtree

# Load iris dataset from Seaborn's data repo
df = pd.read_csv(
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
)
df_hash = _hashobj(df)


def test_log_df(capsys):
    @log_df
    def group_mean(df, grp_col, val_col):
        return df.groupby(grp_col)[val_col].mean().reset_index()

    _ = group_mean(df, "species", "petal_length")
    captured = capsys.readouterr()
    assert "Func group_mean df shape=(3, 2)" in captured.out


def test_maybe(tmp_path):
    p = Path(f"{tmp_path}/test.csv")
    if p.exists():
        p.unlink()

    @maybe(p)
    def f():
        pd.DataFrame(np.random.randn(5, 3)).to_csv(
            tmp_path.joinpath("test.csv"), index=False
        )
        return None

    out = f()
    assert p.exists()
    assert out is None
    out = f()
    assert isinstance(out, pd.DataFrame)
    p.unlink()


# TODO: Add additional tests
def test_disk_cache(tmp_path):
    # Dataframe
    @disk_cache(threshold=5)
    def my_long_func(df, arg, my_kwarg1="hi", my_kwarg2=False):
        sleep(6)
        return df

    cache_dir = Path(".utilz_cache")
    if cache_dir.exists():
        rmtree(str(cache_dir))
    key = cache_dir.joinpath(
        f"my_long_func___arg__2--df__{df_hash}--my_kwarg1__hi--my_kwarg2__false.csv"
    )
    # Remove files from failed tests
    if key.exists():
        key.unlink()
    # First run - should cache and be longer
    tic = dt.datetime.now()
    my_output = my_long_func(df, 2, my_kwarg1="hi", my_kwarg2=False)
    dur1 = dt.datetime.now() - tic
    dur1 = dur1.seconds
    assert my_output.equals(df)
    assert key.exists()
    # Second run - should return cached result and be faster
    tic = dt.datetime.now()
    my_output = my_long_func(df, 2, my_kwarg1="hi", my_kwarg2=False)
    dur2 = dt.datetime.now() - tic
    dur2 = dur2.seconds
    assert dur2 < dur1

    # Remove files from successful tests
    rmtree(str(cache_dir))

    # Other dtypes
    # @disk_cache(threshold=5, verbose=True)
    # def my_long_func(x, arg):
    #     sleep(6)
    #     return x

    # key = "my_long_func_([True], ())"
    # fpath = Path(f"{key}.h5")

    # # Remove files from failed tests
    # if fpath.exists():
    #     fpath.unlink()
    # tic = dt.datetime.now()
    # my_input = [1, 2, 3]
    # _ = my_long_func(my_input, True)
    # dur1 = dt.datetime.now() - tic
    # dur1 = dur1.seconds
    # assert Path(fpath).exists()

    # tic = dt.datetime.now()
    # my_output = my_long_func(my_input, True)
    # dur2 = dt.datetime.now() - tic
    # dur2 = dur2.seconds
    # assert dur2 < dur1
    # assert my_output == my_input
    # # Remove files from successful tests
    # fpath.unlink()
