from pytest import fixture
import pandas as pd


@fixture(scope="module")
def df():
    return pd.read_csv(
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    )
