from utilz.ops import check_random_state, mapcat, filtercat
from utilz.boilerplate import randdf
import numpy as np
import pandas as pd
from time import sleep, time
import pytest
from toolz import pipe


def test_random_state():
    one = check_random_state()
    two = check_random_state()
    assert one.rand(1) != two.rand(1)
    three = check_random_state(999)
    four = check_random_state(999)
    assert three.rand(1) == four.rand(1)


def test_mapcat():

    # Just like map
    out = mapcat(lambda x: x * 2, [1, 2, 3, 4])
    assert out == list(map(lambda x: x * 2, [1, 2, 3, 4]))

    # Currying
    out = pipe([1, 2, 3, 4], mapcat(lambda x: x * 2))
    assert out == list(map(lambda x: x * 2, [1, 2, 3, 4]))

    # Concatenating nested lists
    data = [[1, 2], [3, 4]]
    out = mapcat(None, data)
    assert len(out) == 4

    # Contrived examples as you'd most often just use np funcs directly:

    # Type inference returns same type
    data = np.array(data)
    out = mapcat(np.mean, data)
    assert isinstance(out, np.ndarray)
    assert len(out) == 2
    assert out.ndim == 1

    # If func returns a 1d iterable then concat will be 2d, just like np.array(list of
    # lists)
    out = mapcat(lambda x: np.power(x, 2), data)
    assert isinstance(out, np.ndarray)
    assert out.ndim == 2

    # This is the same as setting axis to 1
    out = mapcat(lambda x: np.power(x, 2), data, axis=1)
    assert out.ndim == 2

    # Axis = 0 will flatten the array to 1d
    out = mapcat(lambda x: np.power(x, 2), data, axis=0)
    assert out.ndim == 1

    # But when concat is false just return a list of numpy arrays
    out = mapcat(lambda x: np.power(x, 2), data, concat=False)
    assert isinstance(out, list)
    assert len(out) == 2
    assert isinstance(out[0], np.ndarray)

    # Passing kwargs to function works
    out = mapcat(np.std, data, func_kwargs={"ddof": 2})
    assert isinstance(out, np.ndarray)
    assert len(out) == 2

    # But they need to be passed as a dict
    with pytest.raises(TypeError):
        out = mapcat(np.std, data, func_kwargs=2)

    # Loading files into a single dataframe
    def load_data(i):
        # simulate dataloading as dfs
        return randdf()

    # Concat pandas
    out = mapcat(load_data, ["file1.txt", "file2.txt", "file3.txt"])
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (30, 3)

    out = mapcat(load_data, ["file1.txt", "file2.txt", "file3.txt"], axis=1)
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (10, 9)


def test_parallel_mapcat():
    def f(x):
        sleep(0.5)
        return x**2

    def f_random(x, random_state=None):
        random_state = check_random_state(random_state)
        sleep(0.5)
        return x + random_state.rand()

    # Running sequentially takes 5s
    start = time()
    out = mapcat(f, range(10), n_jobs=1)
    duration = time() - start
    assert len(out) == 10

    # Running 2 jobs takes less time
    start = time()
    out = mapcat(f, range(10), n_jobs=2)
    par_duration = time() - start
    assert par_duration < duration
    assert len(out) == 10

    # By default if a function to be parallelized handles it's own randomization
    # interally, there should be no issue with repeated values when run in parallel
    out = mapcat(f_random, [1, 1, 1, 1, 1], n_jobs=2)
    assert len(out) == len(set(out))

    # But for reproducibility we can set random_state to a number which will be passed
    # to the func's random_state argument
    out = mapcat(f_random, [1, 1, 1, 1, 1], n_jobs=2, random_state=1)
    out2 = mapcat(f_random, [1, 1, 1, 1, 1], n_jobs=2, random_state=1)
    assert np.allclose(out, out2)

    # But not if it doesn't accept that kwarg
    with pytest.raises(TypeError):
        out = mapcat(f, [1, 1, 1, 1, 1], n_jobs=2, random_state=1)


def test_filtercat():

    # Length 9
    arr = ["aa", "ab", "ac", "ba", "bb", "bc", "ca", "cb", "cc"]
    # Keep anything containing "a"
    assert len(filtercat("a", arr)) == 5
    # Drop anything containing "a"
    assert len(filtercat("a", arr, invert=True)) == 4

    matches, filtered = filtercat("a", arr, invert="split")
    assert len(matches) == 5
    assert len(filtered) == 4

    # Remove multiple
    assert len(filtercat(["aa", "bb", "cc"], arr, invert=True)) == 6

    # Just like filter() if comparison is all False return empty list
    with pytest.raises(AssertionError):
        filtercat(1, arr)
    assert len(filtercat(1, arr, assert_notempty=False)) == 0

    # Currying
    assert len(pipe(arr, filtercat("a"))) == 5

    # Normal filtering
    assert all(filtercat(lambda x: isinstance(x, str), arr))
