from utilz.verbs import rows
from utilz.boilerplate import randdf
from toolz import pipe


def test_rows():

    df = randdf()
    out = pipe(df, rows((0, 5)))
    assert out.shape == (5, 3)

    out = pipe(df, rows([0, 5]))
    assert out.shape == (2, 3)
