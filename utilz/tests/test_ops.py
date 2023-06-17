from utilz import (
    check_random_state,
    map,
    mapcat,
    mapmany,
    mapcompose,
    mapacross,
    mapif,
    mapwith,
    filter,
    pipe,
    spread,
    gather,
    unpack,
    across,
    do,
    iffy,
    append,
    compose,
    curry,
    pop,
    keep,
    discard,
    seq,
    equal,
    fork,
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


def test_map():
    # Just like list comprehension
    out = map(lambda x: x * 2, [1, 2, 3, 4])
    correct = [x * 2 for x in [1, 2, 3, 4]]
    assert out == correct

    # Currying
    out = pipe([1, 2, 3, 4], map(lambda x: x * 2))
    assert out == correct


def test_mapcat():
    # Nested lists are flattened
    data = [[1, 2], [3, 4]]
    out = mapcat(None, data)
    assert len(out) == 4

    # Numpy
    data = np.array(data)

    # If input and return are both numpy arrays then the default concat_axis = None
    # casts using np.array(mapresult)
    # Here 1d func operates on each row of the 2d matrix
    out = mapcat(lambda x: np.power(x, 2), data)
    assert isinstance(out, np.ndarray)
    assert out.ndim == 2

    # But when using regular map just return a list of numpy arrays
    out = map(lambda x: np.power(x, 2), data)
    assert isinstance(out, list)
    assert len(out) == 2
    assert isinstance(out[0], np.ndarray)

    # This is the same as setting axis to 1 manually
    out = mapcat(lambda x: np.power(x, 2), data, concat_axis=1)
    assert out.ndim == 2

    # Setting it to 0, flattens/hstacks the array into a 1d
    out = mapcat(lambda x: np.power(x, 2), data, concat_axis=0)
    assert out.ndim == 1
    assert len(out) == 4

    # Passing kwargs to function works
    out = mapcat(np.std, data, ddof=1)
    assert isinstance(out, np.ndarray)
    assert np.allclose(out, np.std(data, ddof=1, axis=1))
    assert len(out) == 2

    # Loading files into a single dataframe
    def load_data(i):
        # simulate dataloading as dfs
        return randdf()

    # Concat pandas
    out = mapcat(load_data, ["file1.txt", "file2.txt", "file3.txt"])
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (30, 3)

    out = mapcat(load_data, ["file1.txt", "file2.txt", "file3.txt"], concat_axis=1)
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (10, 9)


def test_parallel_map():
    def f(x):
        sleep(0.5)
        return x**2

    def f_random(x, random_state=None):
        random_state = check_random_state(random_state)
        sleep(0.5)
        return x + random_state.rand()

    # Running sequentially takes 5s
    start = time()
    out = map(f, range(20), n_jobs=1)
    duration = time() - start
    assert len(out) == 20

    # Running 2 jobs takes less time
    start = time()
    out = map(f, range(20), n_jobs=2)
    par_duration = time() - start
    assert par_duration < duration
    assert len(out) == 20

    # By default if a function to be parallelized handles it's own randomization
    # interally, there should be no issue with repeated values when run in parallel
    out = map(f_random, [1, 1, 1, 1, 1], n_jobs=2)
    assert len(out) == len(set(out))

    # But for reproducibility we can set random_state to a number which will be passed
    # to the func's random_state argument
    out = map(f_random, [1, 1, 1, 1, 1], n_jobs=2, random_state=1)
    out2 = map(f_random, [1, 1, 1, 1, 1], n_jobs=2, random_state=1)
    assert np.allclose(out, out2)

    # But not if it doesn't accept that kwarg
    with pytest.raises(TypeError):
        out = map(f, [1, 1, 1, 1, 1], n_jobs=2, random_state=1)


def test_mapalts():
    """Just easier shorthands for things we can do with default mapcat"""

    # Map a sequence of functions on after another
    out = pipe(seq(10), mapcompose(lambda x: x**2, np.sqrt))
    correct = pipe(seq(10), map(lambda x: x**2), map(np.sqrt))

    assert np.allclose(out, correct)
    assert np.allclose(out, seq(10))

    # Map multiple functions separately
    out = pipe(seq(10), mapmany(lambda x: x**2, np.sqrt))
    assert len(out) == 10
    assert all([len(e) == 2 for e in out])

    correct = pipe(seq(10), map(lambda e: pipe(e, spread(lambda x: x**2, np.sqrt))))
    assert all(np.allclose(o, c) for o, c in zip(out, correct))

    # Map multiple functions as matched pairs
    out = pipe([2, 4], mapacross(lambda x: x**2, lambda x: x * 2))
    correct = pipe(
        [2, 4],
        lambda tup: (lambda x: x**2, tup[0], lambda x: x * 2, tup[1]),
    )
    assert len(out) == 2
    assert np.allclose(out, [4, 8])

    # Doesnt work if lengths don't match
    with pytest.raises(ValueError):
        pipe([2], mapacross(lambda x: x**2, lambda x: x * 2))

    with pytest.raises(ValueError):
        pipe([2, 4], mapacross(lambda x: x**2))

    # Map a function if a predicate is true
    bigger_5 = lambda x: x > 5
    out = pipe(seq(10), mapif(lambda x: x * 2, bigger_5))
    assert equal(out, [0, 1, 2, 3, 4, 5, 12, 14, 16, 18])

    # Pass a single fixed extra arg
    out = mapwith(lambda fixed, elem: elem + fixed, 5, [1, 2, 3, 4])
    outc = pipe([1, 2, 3, 4], mapwith(lambda x, y: x + y, 5))
    correct = [x + 5 for x in [1, 2, 3, 4]]
    assert out == correct
    assert outc == correct

    # Multiple iterables
    iterme = [1, 2, 3]
    iterwith = [2, 2, 2]
    out = mapwith(lambda x, y: x / y, iterwith, iterme)
    outc = pipe(
        iterme, mapwith(lambda frompipe, iterwith: frompipe / iterwith, iterwith)
    )
    correct = [x / 2 for x in [1, 2, 3]]
    assert out == correct
    assert outc == correct

    # Map around a fixed input
    out = pipe([5, 10, 20], mapwith(lambda e, df: df.shape[0] > e, randdf()))
    assert equal([True, False, False], out)

    df = randdf((20, 3)).assign(Group=["A"] * 5 + ["B"] * 5 + ["C"] * 5 + ["D"] * 5)

    out = pipe(["A", "C"], mapwith(lambda label, df: df.query("Group == @label"), df))
    assert len(out) == 2
    assert out[0].shape[0] == int(df.shape[0] / 4)


def test_filter():
    # Length 9
    arr = ["aa", "ab", "ac", "ba", "bb", "bc", "ca", "cb", "cc"]
    # Keep anything containing "a"
    assert len(filter("a", arr)) == 5
    # Alias
    assert len(keep("a", arr)) == 5
    # Drop anything containing "a"
    assert len(filter("a", arr, invert=True)) == 4
    # Alias
    assert len(discard("a", arr)) == 4

    matches, filtered = filter("a", arr, invert="split")
    assert len(matches) == 5
    assert len(filtered) == 4

    # Remove multiple
    assert len(filter(["aa", "bb", "cc"], arr, invert=True)) == 6

    # Just like filter() if comparison is all False return empty list
    with pytest.raises(AssertionError):
        filter(1, arr)
    assert len(filter(1, arr, assert_notempty=False)) == 0

    # Currying
    assert len(pipe(arr, filter("a"))) == 5

    # Normal filtering
    assert all(filter(lambda x: isinstance(x, str), arr))


def test_pipes_basic():
    df = randdf((20, 3)).assign(Group=["A"] * 5 + ["B"] * 5 + ["C"] * 5 + ["D"] * 5)

    # input -> output
    out = pipe(df, lambda df: df.head())
    assert out.shape == (5, 4)
    out = pipe(df, lambda df: df.head(10), lambda df: df.tail(5))
    assert out.shape == (5, 4)
    assert out.equals(df.iloc[5:10, :])

    # ellipses can be use to terminate a pipe's return value early
    out = pipe(df, lambda df: df.head(10), ..., lambda df: df.tail(5))
    assert out.equals(df.head(10))

    # But there can only be 1
    with pytest.raises(ValueError):
        pipe(
            df,
            lambda df: df.head(10),
            ...,
            lambda df: df.tail(5),
            ...,
            lambda df: df.head(2),
        )

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

    # Use map instead for 1 func
    with pytest.raises(ValueError):
        out = pipe([df, df], mapmany(lambda df: df.head(5)))

    # 2 func-input pairs
    out = pipe([df, df], across(lambda df: df.head(5), lambda df: df.tail(10)))
    assert len(out) == 2
    assert out[0].equals(df.head(5))
    assert out[1].equals(df.tail(10))

    # 2 funcs, i.e. mini-pipe
    out = pipe([df, df], mapcompose(lambda df: df.head(10), lambda df: df.tail(5)))
    assert isinstance(out, list) and len(out) == 2
    assert out[0].equals(out[1])
    assert out[0].equals(df.iloc[5:10, :])

    # not enough funcs
    with pytest.raises(ValueError):
        out = pipe([df, df], across(lambda df: df.head(5)))

    # not enough data
    with pytest.raises(ValueError):
        out = pipe([df], across(lambda df: df.head(5), lambda df: df.tail(2)))

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


def test_iffy():
    bigger_5 = lambda x: x > 5

    # Pass in predicate func and return value
    out = pipe(10, iffy(bigger_5, 2))
    assert out == 2

    # Without if_false returns input by default
    out = pipe(1, iffy(bigger_5, 2))
    assert out == 1

    # Useful to conditionally apply func over iterable when combined with map
    out = pipe(seq(10), map(iffy(bigger_5, lambda x: x * 2)))
    assert equal(out, [0, 1, 2, 3, 4, 5, 12, 14, 16, 18])


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
        ...,
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
        ...,
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
    #       histplot
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
        ...,
        across(
            lambda means: sns.histplot(means),
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
        ...,
        across(
            compose(lambda means: sns.histplot(means), tweak(title="histplot")),
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
        ...,
        across(
            compose(lambda means: sns.histplot(means), tweak(title="histplot")),
            compose(lambda means: sns.boxplot(means), tweak(title="boxplot")),
        ),
        keep=0,
    )
    assert isinstance(a1_mean, pd.Series)

    # Same thing with pop
    a1_mean_with_pop = pipe(
        df,
        lambda df: df.groupby("group"),
        spread(
            lambda dfg: dfg.select("A1").mean(),
            lambda dfg: dfg.select("B1").mean(),
        ),
        pop(0),
        ...,
        compose(lambda means: sns.histplot(means), tweak(title="histplot")),
    )

    assert isinstance(a1_mean_with_pop, pd.Series)
