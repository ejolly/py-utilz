from types import GeneratorType
from utilz.genz import make_gen, combine_gens
import pytest


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
