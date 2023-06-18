import pandas as pd
import numpy as np
from utilz.decorators import timeit, maybe, log, expensive, show
from time import sleep, time
from pathlib import Path
from shutil import rmtree
from joblib import Memory
import pytest
from utilz.io import load
from utilz.shorts import equal


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

    # If function returns nothing, show return input
    @show
    def f(x):
        print("bye")

    out = f("hi")
    assert out == "hi"


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
    temp_file = Path(f"{tmp_path}/test.csv")
    if temp_file.exists():
        temp_file.unlink()

    # Decorate a function that simulates running a computation that saves and returns a
    # dataframe
    @maybe
    def f(shape, **kwargs):
        print("I'm running")
        df = pd.DataFrame(np.random.randn(*shape))
        save_path = kwargs.get("out_file")
        df.to_csv(save_path, index=False)
        return df

    # First run: func executes and saves file
    out = f((5, 3), out_file=temp_file)
    captured = capsys.readouterr()
    assert "I'm running" in captured.out
    assert isinstance(out, pd.DataFrame)
    assert temp_file.exists()

    # Second run: just loads file
    out_loaded = f((5, 3), out_file=temp_file)
    captured = capsys.readouterr()
    assert "I'm running" not in captured.out
    assert f"Loading precomputed result from: {temp_file}" in captured.out
    assert isinstance(out_loaded, pd.DataFrame)
    assert temp_file.exists()
    assert np.allclose(out.to_numpy(), out_loaded.to_numpy())

    # Force a rerun with overwrite=True
    out_rerun = f((5, 3), out_file=temp_file, overwrite=True)
    captured = capsys.readouterr()
    assert "I'm running" in captured.out
    assert isinstance(out_rerun, pd.DataFrame)
    assert temp_file.exists()
    assert not np.allclose(out_loaded.to_numpy(), out_rerun.to_numpy())

    # Missing out_file as a kwarg to decorated function raises error
    with pytest.raises(ValueError):
        f((5, 3))

    # Clean up
    temp_file.unlink()

    # Test with list of files
    temp_dir = Path(f"{tmp_path}/myfiles")

    @maybe
    def f(shape, **kwargs):
        save_folder = kwargs.get("out_file")
        save_folder.mkdir()
        print("I'm running")
        out = []
        for i in range(3):
            df = pd.DataFrame(np.random.randn(*shape))
            df.to_csv(save_folder / f"{i}.csv", index=False)
            out.append(df)
        return out

    # First run: func executes and saves multiple files to dir
    out = f((5, 3), out_file=temp_dir)
    captured = capsys.readouterr()
    assert "I'm running" in captured.out
    assert isinstance(out, list)
    assert isinstance(out[0], pd.DataFrame)
    assert any(temp_dir.iterdir())

    # Second run: just loads files
    out_loaded = f((5, 3), out_file=temp_dir)
    captured = capsys.readouterr()
    assert "I'm running" not in captured.out
    assert f"Loading precomputed result from: {temp_dir}" in captured.out
    assert isinstance(out_loaded, list)
    assert len(list(temp_dir.iterdir()))
    assert all(
        np.allclose(one.to_numpy(), two.to_numpy()) for one, two in zip(out, out_loaded)
    )

    # Passing custom loader works, in this case no-op to just get file names
    custom_loader = lambda x: x

    maybe_out_files = f((5, 3), out_file=temp_dir, loader_func=custom_loader)

    # Equivalent to loading file names directly
    out_files = load(temp_dir, loader_func=custom_loader)

    assert equal(maybe_out_files, out_files)

    rmtree(temp_dir)


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
