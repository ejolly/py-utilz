from utilz.ops import (
    check_random_state,
    mapcat,
    filtercat,
    pipe,
    spread,
    gather,
    unpack,
    separate,
    do,
    ifelse,
    append,
    compose,
    curry,
    pop,
)
from utilz.plot import tweak
from utilz.boilerplate import randdf
import numpy as np
import pandas as pd
from time import sleep, time
import pytest
import seaborn as sns


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


def test_pipes_basic():

    df = randdf((20, 3)).assign(Group=["A"] * 5 + ["B"] * 5 + ["C"] * 5 + ["D"] * 5)

    # input -> output
    out = pipe(df, lambda df: df.head())
    assert out.shape == (5, 4)
    out = pipe(df, lambda df: df.head(10), lambda df: df.tail(5))
    assert out.shape == (5, 4)
    assert out.equals(df.iloc[5:10, :])

    # APPEND (simplified one2many)
    # simplified version of spread when you know you need result from previous step and
    # just 1 other thing
    # input -> (input, output)
    out = pipe(df, append(lambda df: df.head()))
    assert isinstance(out, tuple)
    assert out[0].equals(df) and out[1].equals(df.head())

    # # multiple in a row:
    # # input -> (input, output) -> (input, output, output2)
    out = pipe(df, append(lambda df: df.head()), append(lambda df: df.tail()))
    assert len(out) == 3
    assert not out[1].equals(out[2])
    assert out[0].equals(df)

    # We can access all previously appended values by passing in a func that takes more
    # than 1 arg
    out = pipe(
        df,
        append(lambda df: df.head()),
        append(lambda df, head: df.tail().iloc[:, :2] + head.iloc[:, :2]),
    )
    assert len(out) == 3
    assert out[0].equals(df)
    assert out[1].equals(df.head())
    assert out[2].equals(df.tail().iloc[:, :2] + df.head().iloc[:, :2])

    # SEPARATE (many2many)
    # (input1, input2) -> (output1, output2)

    # 1 func
    out = pipe([df, df], separate(lambda df: df.head(5), match=False))
    assert isinstance(out, tuple) and len(out) == 2  # 2x1
    assert out[0].equals(out[1])
    assert out[0].equals(df.head(5))

    # 2 func-input pairs
    out = pipe([df, df], separate(lambda df: df.head(5), lambda df: df.tail(10)))
    assert len(out) == 2
    assert out[0].equals(df.head(5))
    assert out[1].equals(df.tail(10))

    # 2 funcs, i.e. mini-pipe
    out = pipe(
        [df, df], separate(lambda df: df.head(10), lambda df: df.tail(5), match=False)
    )
    assert isinstance(out, tuple) and len(out) == 2
    assert out[0].equals(out[1])
    assert out[0].equals(df.iloc[5:10, :])

    # mismatch
    with pytest.raises(ValueError):
        out = pipe([df, df], separate(lambda df: df.head(5)))

    # not enough data
    with pytest.raises(TypeError):
        out = pipe(df, separate(lambda df: df.head(5)))

    # SPREAD (one2many)
    # input -> (input, input, input)
    out = pipe(df, spread(3))
    assert isinstance(out, tuple)
    assert len(out) == 3
    assert all(df.equals(ddf) for ddf in out)
    # make sure they're copies
    assert all(df is not ddf for ddf in out)

    # input -> (output1, output2)
    out = pipe(df, spread(lambda df: df.head(), lambda df: df.mean()))
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert out[0].shape == (5, 4)
    assert out[1].shape == (3,)

    # spread expects more than 1 func, should use do or inline instead
    with pytest.raises(ValueError):
        out = pipe(df, spread(lambda df: df.head()))

    # GATHER (many2one)
    # (input1, input2) -> output
    out = pipe([df, df], gather(lambda df1, df2: df1 + df2))
    assert out.equals(df + df)
    # Test alias
    out = pipe([df, df], unpack(lambda df1, df2: df1 + df2))
    assert out.equals(df + df)
    # same thing just nicer semantics gather
    out = pipe([df, df], lambda dfs: dfs[0] + dfs[1])
    assert out.equals(df + df)

    # error not enough data
    with pytest.raises(TypeError):
        pipe(df, gather(lambda df: df.head()))


def test_do():
    df = randdf()

    # Do is equivalent to a pipe with a single function, but the signature is reversed
    # because do is also curried and can be used within a pipe:
    # pipe(data, func)
    # do(func, data)
    out = do(lambda df: df.head(), df)
    assert out.equals(df.head())
    assert out.equals(pipe(df, lambda df: df.head()))

    out = do("head", df)
    assert out.equals(df.head())

    out = pipe(df, do("head"))
    assert out.equals(df.head())

    # With no currying can omit kwarg names and they will be passed to func/method in
    # positional order
    out = do("head", df, 10)
    assert out.equals(df.head(10))

    # With currying you must use kwargs
    with pytest.raises(AttributeError):
        out = pipe(df, do("head", 10))

    # like this
    out = pipe(df, do("head", n=10))
    assert out.equals(df.head(10))


def test_ifelse():

    x = 10

    # Can call with boolean expressions
    y = ifelse(x > 10, x + 1, x - 1)
    assert y < x
    y = ifelse(x > 1, x + 1, x)
    assert y > x

    # Can call as eval string with special variable 'data'
    y = ifelse("data > 10", x + 1, x - 1, x)
    assert y < x

    # Can call as func and then use whatever name
    out = ifelse(lambda e: e * 2 > 20, "yes", "no", x)
    assert out == "no"

    # Can return them too
    out = ifelse(lambda e: e * 2 > 10, lambda e: e + 10, "no", x)
    assert out == 20

    # In pipes, ifelse is curried
    out = pipe(10, ifelse("data > 0", "yes", "no"))
    assert out == "yes"

    # Same thing
    out = pipe(10, ifelse(lambda data: data > 0, "yes", "no"))
    assert out == "yes"

    # But more powerful
    out = pipe([10, 20], ifelse(lambda list: list[0] > list[1], "yes", "no"))
    assert out == "no"

    # Combine with gather to name args
    out = pipe([10, 20], gather(lambda a, b: ifelse(a > b, "yes", "no")))
    assert out == "no"


def test_pop():
    # Pop is for shrinking outputs to other funcs
    # Keep is pruning the final outputs of a complicated pipe

    df = randdf()
    out = pipe(df, append(lambda df: df.mean()), pop(1))
    assert out.equals(df)

    # Equivalent using keep kwarg
    out2 = pipe(df, append(lambda df: df.mean()), keep=0)
    assert out2.equals(out)

    out = pipe(df, append(lambda df: df.mean()), append(lambda df: df.head()), pop(0))
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert out[-1].equals(df.head())

    # Equivalent using keep kwarg
    out2 = pipe(
        df, append(lambda df: df.mean()), append(lambda df: df.head()), keep=(1, 2)
    )
    assert isinstance(out2, tuple)
    assert out2[-1].equals(df.head())


def test_pipes_advanced():
    df = randdf((20, 3), groups={"condition": 2, "group": 4})

    # df ->
    # (df, labels) ->
    #   barplot
    #   regplot
    # Returns: (df, labels)
    out = pipe(
        df,
        append(lambda df: df["group"].unique()[::-1]),
        spread(
            lambda tup: sns.barplot(
                x="group",
                y="A1",
                hue="condition",
                order=tup[1],
                data=tup[0],
            ),
            lambda tup: sns.regplot(x="A1", y="B1", data=tup[0]),
        ),
    )
    assert len(out) == 2
    assert out[0].equals(df)
    assert all(out[1] == df["group"].unique()[::-1])

    # Same as above but we stick a gather in their to more easily unpack the tuple
    out = pipe(
        df,
        append(lambda df: df["group"].unique()[::-1]),
        spread(
            gather(
                lambda data, order: sns.barplot(
                    x="group", y="A1", hue="condition", order=order, data=data
                ),
            ),
            gather(
                lambda data, _: sns.regplot(x="A1", y="B1", data=data),
            ),
        ),
    )
    assert len(out) == 2
    assert out[0].equals(df)
    assert all(out[1] == df["group"].unique()[::-1])

    # df ->
    # df-grouped ->
    #   A1 mean ->
    #   B1 mean ->
    # pd.concat
    # Returns: df of (A1 mean, B1 mean)
    out = pipe(
        df,
        lambda df: df.groupby("group"),
        spread(
            lambda dfg: dfg.select("A1").mean(),
            lambda dfg: dfg.select("B1").mean(),
        ),
        curry(pd.concat, axis=1),
    )
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (4, 2)

    # df ->
    # df-grouped ->
    #   A1 mean ->
    #       distplot
    #   B1 mean ->
    #       boxplot
    #
    # Returns: tuple (A1 mean, B1 mean)
    out = pipe(
        df,
        lambda df: df.groupby("group"),
        spread(
            lambda dfg: dfg.select("A1").mean(),
            lambda dfg: dfg.select("B1").mean(),
        ),
        separate(
            lambda means: sns.distplot(means),
            lambda means: sns.boxplot(means),
        ),
        debug=True,
    )
    assert len(out) == 3  # 3 steps in pipe
    assert isinstance(out[-1], tuple)  # plots
    assert isinstance(out[-2], tuple)  # series
    assert isinstance(out[-3], pd.core.groupby.generic.DataFrameGroupBy)

    # Same as above but with a compose thrown in for extra complexity.
    # Without debug=True we can unpack the tuple
    out1, out2 = pipe(
        df,
        lambda df: df.groupby("group"),
        spread(
            lambda dfg: dfg.select("A1").mean(),
            lambda dfg: dfg.select("B1").mean(),
        ),
        separate(
            compose(lambda means: sns.distplot(means), tweak(title="distplot")),
            compose(lambda means: sns.boxplot(means), tweak(title="boxplot")),
        ),
    )
    assert isinstance(out1, pd.Series)
    assert isinstance(out2, pd.Series)

    # Same but this time we pop off the last value and only get 1 return
    a1_mean = pipe(
        df,
        lambda df: df.groupby("group"),
        spread(
            lambda dfg: dfg.select("A1").mean(),
            lambda dfg: dfg.select("B1").mean(),
        ),
        separate(
            compose(lambda means: sns.distplot(means), tweak(title="distplot")),
            compose(lambda means: sns.boxplot(means), tweak(title="boxplot")),
        ),
        keep=0,
    )
    assert isinstance(a1_mean, pd.Series)

    # NOTE: THIS DOESN"T WORK BY DESIGN
    # Pop can only shrink outputs from the previous step
    # in this case thats a tuple of plots which are discarded anyway
    # so the real output (from spread) is unaffected
    a1_mean_with_pop = pipe(
        df,
        lambda df: df.groupby("group"),
        spread(
            lambda dfg: dfg.select("A1").mean(),
            lambda dfg: dfg.select("B1").mean(),
        ),
        separate(
            compose(lambda means: sns.distplot(means), tweak(title="distplot")),
            compose(lambda means: sns.boxplot(means), tweak(title="boxplot")),
        ),
        pop(0),
    )
    assert isinstance(a1_mean_with_pop, tuple)
    assert len(a1_mean_with_pop) == 2
    assert a1_mean_with_pop[0].equals(a1_mean)
