# flake8: noqa
import pandas as pd
from utilz.pipe import Pipe

# Intialize pipe
o = Pipe()


def test_pipe(capsys):
    # Load iris dataset from Seaborn's data repo
    df = pd.read_csv(
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    )

    out = df >> o >> (lambda df: df * 2)
    assert out.shape == df.shape

    head = df >> o >> "head"
    assert all(head == df.head())

    mean = df >> o >> ("mean", 1)
    assert all(mean == df.mean(1))

    melted = (
        df >> o >> (pd.melt, {"id_vars": "species"}, {"value_vars": "petal_length"})
    )
    pd_melted = pd.melt(df, id_vars="species", value_vars="petal_length")
    assert all(melted == pd_melted)

    result = "name" >> o >> ",".join
    assert result == "n,a,m,e"

    result = "  name  " >> o >> "strip" >> o >> ",".join
    assert result == "n,a,m,e"

    # smoke tests
    [1, 2, 3] >> o >> print

    captured = capsys.readouterr()

    assert "[1, 2, 3]" in captured.out

    "hi" >> o >> print

    captured = capsys.readouterr()
    assert "hi" in captured.out
