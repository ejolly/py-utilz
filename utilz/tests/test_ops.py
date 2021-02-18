from math import sqrt
from utilz.ops import prep, random_seed, pmap


# TODO pmap has some issues with cannot unpack non-iterable function object:
def test_pmap(capsys):
    pass


def test_prep(capsys):
    def mysqrt(idx=None):
        """math.sqrt modified to handle idx"""
        return sqrt(idx)

    def mysqrt_missing_seed(idx=None, double=False):
        """math.sqrt modified to handle: idx and kwarg"""
        if double:
            return sqrt(idx) * 2
        else:
            return sqrt(idx)

    def mysqrt_seed(idx=None, double=False, seed=None):
        """math.sqrt modified to handle: idx, kwarg, and seed"""
        new_seed = random_seed(seed)
        elem = new_seed.choice(idx + 1, 1)
        if double:
            return sqrt(elem) * 2
        else:
            return sqrt(elem)

    # User forget idx
    try:
        out = prep(sqrt)
    except TypeError as e:
        assert "no keyword arguments" in str(e)

    # Func that only takes idx; runs serially
    out = prep(mysqrt, n_iter=10000, n_jobs=1, verbose=10)
    captured = capsys.readouterr()
    assert "Parallel(n_jobs=1)]: Done 10000 out of 10000 | elapsed:" in captured.err
    assert len(out) == 10000

    # Func that only takes idx; parallel
    out = prep(mysqrt, n_iter=100000, n_jobs=-1, verbose=10)
    captured = capsys.readouterr()
    assert "Parallel(n_jobs=-1)]: Done 100000 out of 100000 | elapsed:" in captured.err
    assert len(out) == 100000

    # Func that only takes args; parallel
    myfunc = lambda x: x
    out = prep(myfunc, [10], n_iter=100000, n_jobs=-1, loop_idx=False, verbose=10)
    captured = capsys.readouterr()
    assert "Parallel(n_jobs=-1)]: Done 100000 out of 100000 | elapsed:" in captured.err
    assert sum(out) == 10 * 100000

    # Func that takes kwargs idx, but user forgets seed; parallel
    try:
        my_out = prep(
            mysqrt_missing_seed,
            {"double": True},
            n_iter=100000,
            n_jobs=-1,
            verbose=10,
            loop_random_seed=True,
        )
    except TypeError as e:
        assert "unexpected keyword argument" in str(e)

    # Func that takes kwargs, idx, and seed; parallel
    my_out = prep(
        mysqrt_seed,
        {"double": True},
        n_iter=100000,
        n_jobs=-1,
        verbose=10,
        loop_random_seed=True,
    )
    captured = capsys.readouterr()
    assert "Parallel(n_jobs=-1)]: Done 100000 out of 100000 | elapsed:" in captured.err
    assert sum(my_out) > sum(out)
    my_out_two = prep(
        mysqrt_seed,
        {"double": True},
        n_iter=100000,
        n_jobs=-1,
        verbose=10,
        loop_random_seed=True,
    )
    # Check seed is actually different
    assert [a != b for a, b in zip(my_out, my_out_two)]
