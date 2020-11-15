from utilz.guards import same_shape
import pandas as pd


df = pd.read_csv(
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
)


def test_same_shape():
    @same_shape("species")
    def groupem(df):
        return df.groupby("species").sepal_length.mean()

    grouped = groupem(df)
    print(grouped.shape)
    print(grouped)
