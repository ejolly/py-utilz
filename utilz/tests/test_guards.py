import pandas as pd
from utilz.guards import log_df


def test_log_df(capsys):
    # Load iris dataset from Seaborn's data repo
    df = pd.read_csv(
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    )

    @log_df
    def group_mean(df, grp_col, val_col):
        return df.groupby(grp_col)[val_col].mean().reset_index()

    _ = group_mean(df, 'species', 'petal_length')
    captured = capsys.readouterr()
    assert 'Func group_mean df shape=(3, 2)' in captured.out