import pandas as pd
import numpy as np
from utilz.decorators import timeit, maybe, log, expensive, show
from time import sleep, time
from pathlib import Path
from shutil import rmtree
from joblib import Memory


def test_show(capsys, df):
    @show
    def f(x):
        return x

    _ = f(10)
    captured = capsys.readouterr()
    assert "10" in captured.out

    @show
    def ff(df):
        return df

    _ = ff(df)
    captured = capsys.readouterr()
    dfs = df.head().to_string() + "\n"
    assert captured.out == dfs


def test_log(df, capsys):
    @log
    def group_mean(df, grp_col, val_col):
        return df.groupby(grp_col)[val_col].mean().reset_index()

    _ = group_mean(df, "species", "petal_length")
    captured = capsys.readouterr()
    assert "group_mean, (3, 2), df" in captured.out

    @log
    def empty(arr):
        return arr

    _ = empty(df.to_numpy())
    captured = capsys.readouterr()
    assert "empty, (150, 5), np" in captured.out

    empty(df.to_dict())
    captured = capsys.readouterr()
    assert "empty, 5, {}" in captured.out

    empty([df])
    captured = capsys.readouterr()
    assert "empty, 1, []" in captured.out


def test_timeit(df, capsys):
    @timeit
    def myfunc(df):
        sleep(3)
        return df

    _ = myfunc(df)
    captured = capsys.readouterr()
    assert "myfunc, took" in captured.out
    assert "3" in captured.out


def test_maybe(tmp_path, capsys):
    p = Path(f"{tmp_path}/test.csv")
    if p.exists():
        p.unlink()

    # Decorate a function that simulates running a computation that saves and returns a
    # dataframe
    @maybe(p)
    def f(save_to):
        print("I'm running")
        df = pd.DataFrame(np.random.randn(5, 3))
        df.to_csv(tmp_path.joinpath(save_to), index=False)
        return df

    # First run: func executes and saves file
    out = f("test.csv")
    captured = capsys.readouterr()
    assert "I'm running" in captured.out
    assert isinstance(out, pd.DataFrame)
    assert p.exists()

    # Second run: just loads file
    out_loaded = f("test.csv")
    captured = capsys.readouterr()
    assert "I'm running" not in captured.out
    assert "Exists: loading previously saved file" in captured.out
    assert isinstance(out_loaded, pd.DataFrame)
    assert p.exists()
    assert np.allclose(out.to_numpy(), out_loaded.to_numpy())

    # Force reruns by adding kwargs to the decorator
    @maybe(p, force=True)
    def f(save_to):
        print("I'm running")
        df = pd.DataFrame(np.random.randn(5, 3))
        df.to_csv(tmp_path.joinpath(save_to), index=False)
        return df

    # Third run: func reruns and overwrites files
    out_rerun = f("test.csv")
    captured = capsys.readouterr()
    assert "I'm running" in captured.out
    assert isinstance(out_rerun, pd.DataFrame)
    assert p.exists()
    assert not np.allclose(out.to_numpy(), out_rerun.to_numpy())

    # Clean up
    p.unlink()

    # Test with list of files
    p = Path(f"{tmp_path}/myfiles")
    p.mkdir()

    @maybe(p)
    def f(p):
        print("I'm running")
        out_files = []
        for i in range(3):
            df = pd.DataFrame(np.random.randn(5, 3))
            df.to_csv(p / f"{i}.csv", index=False)
            out_files.append(df)
        return out_files

    out_files = f(p)
    captured = capsys.readouterr()
    assert "I'm running" in captured.out
    assert isinstance(out_files, list)
    assert isinstance(out_files[0], pd.DataFrame)

    out_files = f(p)
    captured = capsys.readouterr()
    assert "I'm running" not in captured.out
    assert "Exists: loading previously saved file" in captured.out
    assert isinstance(out_files, list)
    assert isinstance(out_files[0], Path)


def test_expensive(df, capsys):

    memory = Memory("./cachedir")
    memory.clear()

    @expensive()
    def myfunc(df):
        print("Computing...")
        sleep(3)
        return df

    # First run
    start = time()
    _ = myfunc(df)
    duration = time() - start
    assert 3 <= duration < 4
    captured = capsys.readouterr()
    assert "Computing..." in captured.out

    # Second run
    start = time()
    _ = myfunc(df)
    duration = time() - start
    assert duration < 1
    captured = capsys.readouterr()
    assert "Computing..." not in captured.out

    # Third forced rerurun
    @expensive(force=True)
    def myfunc(df):
        print("Computing...")
        sleep(3)
        return df

    start = time()
    _ = myfunc(df)
    duration = time() - start
    assert 3 <= duration < 4
    captured = capsys.readouterr()
    assert "Computing..." in captured.out

    # Clean up
    rmtree("./cachedir")
