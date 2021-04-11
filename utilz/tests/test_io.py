from utilz.io import load, save
import pytest
import numpy as np
import pandas as pd
import deepdish as dd
import pickle
import json
from pathlib import Path


@pytest.fixture
def setup_data(tmp_path: Path):
    """
    Creates several test data files:
    arr.txt, arr.gz, mydicts.json, mdictlist.json, h5.h5, pickle.p, df.csv, df.h5, df.hd5f
    """

    arr = np.random.randn(10)
    np.savetxt(tmp_path.joinpath("arr.txt"), arr)
    np.savetxt(tmp_path.joinpath("arr.gz"), arr)

    tmp_path.joinpath("txt.txt").write_text("hello world\nhello world")

    mydict = {"name": "Bob", "age": 1}
    with tmp_path.joinpath("mydict.json").open("w") as file_handle:
        json.dump(mydict, file_handle)
    tmp_path.joinpath("mydicts.json").write_text(json.dumps(mydict))

    mydictlist = [{"name": "Goten", "age": 1}, {"name": "Mr.Popo", "age": 999}]
    with tmp_path.joinpath("mydictlist.json").open("w") as file_handle:
        json.dump(mydictlist, file_handle)
    tmp_path.joinpath("mydictlists.json").write_text(json.dumps(mydictlist))

    dd.io.save(str(tmp_path.joinpath("h5.h5")), mydict)

    with tmp_path.joinpath("pickle.p").open("wb") as file_handle:
        pickle.dump(mydictlist, file_handle)

    df = pd.DataFrame(np.random.randn(10, 3), columns=["a", "b", "c"])
    df.to_csv(tmp_path.joinpath("df.csv"), index=False)
    df.to_hdf(tmp_path.joinpath("df.hdf5"), key="data")
    df.to_hdf(tmp_path.joinpath("df.h5"), key="custom_key")


@pytest.mark.usefixtures("setup_data")
def test_load(tmp_path: Path):

    # Load csv
    out = load(tmp_path.joinpath("df.csv"))
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (10, 3)

    # Load hdf5
    out = load(tmp_path.joinpath("df.hdf5"))
    assert isinstance(out, dict)
    out = load(tmp_path.joinpath("df.hdf5"), as_df=True)
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (10, 3)
    out = load(tmp_path.joinpath("df.h5"), as_df=True, h5_key="custom_key")
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (10, 3)
    out = load(tmp_path.joinpath("h5.h5"))
    assert isinstance(out, dict)
    assert "name" in out.keys()

    # load txt
    out = load(tmp_path.joinpath("arr.txt"), as_arr=True)
    assert isinstance(out, np.ndarray)
    assert len(out) == 10
    out = load(tmp_path.joinpath("txt.txt"))
    assert isinstance(out, list)
    assert len(out) == 2
    out = load(tmp_path.joinpath("txt.txt"), as_str=True)
    assert isinstance(out, str)
    assert out == "hello world\nhello world"

    # load pickle
    out = load(tmp_path.joinpath("pickle.p"))
    assert isinstance(out, list)
    assert len(out) == 2
    assert isinstance(out[0], dict)
    assert "name" in out[0].keys()

    # load json
    out = load(tmp_path.joinpath("mydict.json"))
    assert isinstance(out, dict)
    assert "name" in out.keys()
    out = load(tmp_path.joinpath("mydicts.json"), json_str=True)
    assert isinstance(out, dict)
    assert "name" in out
    out = load(tmp_path.joinpath("mydictlist.json"))
    assert isinstance(out, list)
    assert len(out) == 2
    assert "name" in out[0].keys()
    out = load(tmp_path.joinpath("mydictlists.json"), json_str=True)
    assert isinstance(out, list)
    assert len(out) == 2
    assert "name" in out[0].keys()

    # load gz
    out = load(tmp_path.joinpath("arr.gz"))
    assert isinstance(out, np.ndarray)
    assert len(out) == 10

    # incorrect type
    with pytest.raises(TypeError):
        out = load(10)

    with pytest.raises(TypeError):
        out = load("doesnotexist")

    with pytest.raises(FileNotFoundError):
        out = load("no.txt")


@pytest.mark.usefixtures("setup_data")
def test_save(tmp_path: Path):

    names = [
        "file.csv",
        "file.txt",
        "file.h5",
        "file.hdf5",
        "file.npy",
        "file.p",
        "file.pickle",
    ]

    # With dataframe
    out = load(tmp_path.joinpath("df.csv"))
    for name in names:
        out_path = tmp_path.joinpath(name)
        if name.endswith(".npy"):
            save(out_path, out.to_numpy())
        elif name.endswith(".txt") or name.endswith(".npy"):
            with pytest.raises(TypeError):
                save(tmp_path.joinpath(name), out)
        else:
            save(out_path, out, use_method="to_csv", index=False)
