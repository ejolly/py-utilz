from math import sqrt
from utilz.ops import ploop


def test_ploop(capsys):

    # Func that only takes idx; serial
    out = ploop(sqrt, n_iter=10000, n_jobs=1)
    captured = capsys.readouterr()
    assert "Parallel(n_jobs=1)]: Done 10000 out of 10000 | elapsed:" in captured.err

    # Func that only takes idx; parallel
    out = ploop(sqrt, n_iter=100000, n_jobs=-1)
    captured = capsys.readouterr()
    assert "Parallel(n_jobs=-1)]: Done 100000 out of 100000 | elapsed:" in captured.err

    # Func that only takes args; parallel
    myfunc = lambda x: x
    out = ploop(myfunc, [10], n_iter=100000, n_jobs=-1, loop_idx_available=False)
    captured = capsys.readouterr()
    assert "Parallel(n_jobs=-1)]: Done 100000 out of 100000 | elapsed:" in captured.err
    assert sum(out) == 10 * 100000

    # Func that takes args and idx; parallel
    def mysqrt(double, elem):
        if double:
            return sqrt(elem) * 2
        else:
            return sqrt(elem)

    my_out = ploop(mysqrt, [True], n_iter=100000, n_jobs=-1)
    captured = capsys.readouterr()
    assert "Parallel(n_jobs=-1)]: Done 100000 out of 100000 | elapsed:" in captured.err
    assert sum(my_out) > sum(out)

