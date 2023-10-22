from types import GeneratorType
from utilz.genz import make_gen, combine_gens
from utilz.maps import map
from timeit import timeit
from time import sleep, time


def test_make_gen():
    l = list(range(10))
    g = make_gen(l)
    assert isinstance(g, GeneratorType)


def test_combine_gens():
    l1 = list(range(10))
    l2 = list(range(5))
    l3 = list(range(3))
    l4 = list(range(20))
    out = list(combine_gens(l1, l2, l3, l4))
    assert len(out) == len(l1) * len(l2) * len(l3) * len(l4)
    assert out[-1] == (9, 4, 2, 19)


def test_parallel_gens():
    # Note combining generators means function
    # should handle variable args and unpack them
    # as needed
    def sleepy_sum(*args):
        # a, b = args
        sleep(1)
        return sum(*args)

    l1 = list(range(3))
    l2 = list(range(2))

    start = time()
    single = map(sleepy_sum, combine_gens(l1, l2))
    single_time = time() - start

    start = time()
    par = map(sleepy_sum, combine_gens(l1, l2), n_jobs=2)
    par_time = time() - start

    assert single == par
    assert single_time > par_time
