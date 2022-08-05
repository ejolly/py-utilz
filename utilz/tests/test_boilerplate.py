import pandas as pd
import numpy as np
from utilz.boilerplate import randdf


def test_randdf():
    df = randdf()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (10, 3)
    assert all((a == b for a, b in zip(df.columns, ["A1", "B1", "C1"])))

    # Custom columns
    five_col = ["A", "B", "C", "D", "E"]

    df = randdf(
        size=(20, 5),
        func=np.random.beta,
        columns=five_col,
        a=2,
        b=8.2,
    )
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (20, 5)
    assert all((a == b for a, b in zip(df.columns, five_col)))
