import numpy as np
import pandas as pd
from utilz.data import Box
from utilz import randdf, equal, map, seq
import pytest


def test_box():
    # Dataframe data
    df_data = [
        randdf((20, 3), groups={"condition": 2, "group": 4}),
        randdf((20, 3), groups={"condition": 2, "group": 4}),
    ]

    box = Box(df_data)

    # We can get the data in a box by slicing it or using .contents()
    assert box[0].equals(df_data[0])
    assert all([x.equals(y) for x, y in zip(box[:], df_data)])
    assert all([x.equals(y) for x, y in zip(box.contents(), df_data)])

    # By default boxes are transparent and always return the contents of an operation

    # Method access
    out = box.head()
    assert isinstance(out, list)

    correct = map(lambda x: x.head(), df_data)
    assert all([x.equals(y) for x, y in zip(out, correct)])

    # Attribute access
    out = box.shape
    assert isinstance(out, list)

    correct = map(lambda x: x.shape, df_data)
    assert all(map(lambda tup: equal(*tup), zip(out, correct)))

    # Numpy arrays
    data = [np.random.randn(10) for i in range(10)]
    box = Box(data)

    # Method access
    out = box.mean()
    assert isinstance(out, list)

    correct = map(lambda x: x.mean(), data)
    assert all(map(lambda tup: equal(*tup), zip(out, correct)))

    # Attribute access
    out = box.shape
    assert isinstance(out, list)

    correct = map(lambda x: x.shape, data)
    assert all(map(lambda tup: equal(*tup), zip(out, correct)))

    # Opaque boxes return a new box who's contents can be accessed using .contents()
    black_box = Box(df_data, transparent=False)
    assert isinstance(black_box.head(), Box)
    assert isinstance(black_box.head().contents(), list)
    # slice notation works too
    assert isinstance(black_box.head()[:], list)

    # Opaque boxes are useful for method chaining on underlying data
    out = black_box.groupby("group").mean().contents()
    assert isinstance(out, list)

    # Doesn't work cause box is transparent
    with pytest.raises(AttributeError):
        box = Box(df_data)
        box.group_by("group").mean()

    # We can change transparency on the fly
    box.set_transparent(False)
    compare = box.groupby("group").mean().contents()
    assert all([x.equals(y) for x, y in zip(out, compare)])

    # Applying arbitrary functions to box elements
    data = seq(10)
    correct = seq(1, 11)
    box = Box(data)

    # By default map returns the result of the operation just like calling other
    # attributes or methods
    result = box.map(lambda x: x + 1)
    assert isinstance(result, list)
    assert equal(result, correct)

    # Maps respect box transparency which is useful for method chaing
    box = Box(df_data, transparent=False)
    result = box.map(lambda x: x["A1"] + 1).head().contents()
    assert isinstance(result, list)

    # Map operations can also happen inplace which will change box contents without
    # returning anything.
    box = Box(data)
    box.map(lambda x: x + 1, inplace=True)
    assert equal(box.contents(), correct)

    # In place doesn't care if the box is transparent or not.
    box = Box(data, transparent=False)
    box.map(lambda x: x + 1, inplace=True)
    assert equal(box.contents(), correct)
