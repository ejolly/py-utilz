from utilz.io import load, crawl
import pytest
import numpy as np
import pandas as pd
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
    np.save(tmp_path.joinpath("arr.npy"), arr)
    np.save(tmp_path.joinpath("arr.npz"), arr)

    tmp_path.joinpath("txt.txt").write_text("hello world\nhello world")

    mydict = {"name": "Bob", "age": 1}
    with tmp_path.joinpath("mydict.json").open("w") as file_handle:
        json.dump(mydict, file_handle)
    tmp_path.joinpath("mydicts.json").write_text(json.dumps(mydict))

    mydictlist = [{"name": "Goten", "age": 1}, {"name": "Mr.Popo", "age": 999}]
    with tmp_path.joinpath("mydictlist.json").open("w") as file_handle:
        json.dump(mydictlist, file_handle)
    tmp_path.joinpath("mydictlists.json").write_text(json.dumps(mydictlist))

    with tmp_path.joinpath("pickle.p").open("wb") as file_handle:
        pickle.dump(mydictlist, file_handle)

    df = pd.DataFrame(np.random.randn(10, 3), columns=["a", "b", "c"])
    df.to_csv(tmp_path.joinpath("df.csv"), index=False)


@pytest.mark.usefixtures("setup_data")
def test_load(tmp_path: Path):
    # Load csv
    out = load(tmp_path.joinpath("df.csv"))
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (10, 3)

    # load txt
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
    out = load(tmp_path.joinpath("mydicts.json"), as_str=True)
    assert isinstance(out, dict)
    assert "name" in out
    out = load(tmp_path.joinpath("mydictlist.json"))
    assert isinstance(out, list)
    assert len(out) == 2
    assert "name" in out[0].keys()
    out = load(tmp_path.joinpath("mydictlists.json"), as_str=True)
    assert isinstance(out, list)
    assert len(out) == 2
    assert "name" in out[0].keys()

    # load numpy
    out = load(tmp_path.joinpath("arr.txt"), as_arr=True)
    assert isinstance(out, np.ndarray)
    assert len(out) == 10

    out = load(tmp_path.joinpath("arr.npy"))
    assert isinstance(out, np.ndarray)
    assert len(out) == 10

    # Custom loader function
    def custom_loader(file):
        with Path(file).open() as file_handle:
            out = file_handle.readlines()
        return out

    out = load(".gitignore", verbose=True, loader_func=custom_loader)

    # Attemp everything else as text, but with no custom loader issue warning
    with pytest.warns(UserWarning, match="not supported"):
        out = load(".gitignore", verbose=True)

    # incorrect type
    with pytest.raises(TypeError):
        out = load(10)

    # missing file
    with pytest.raises(FileNotFoundError):
        out = load("no.txt")

    # empty file
    empty = Path("empty")
    empty.touch()
    with pytest.raises(AssertionError):
        out = load("empty")

    # No error
    out = load("empty", assert_notempty=False)
    empty.unlink()


def test_crawl():
    # TODO: test ignore arg more thoroughly
    project_root = Path(__file__).parent.parent.parent
    out = crawl(project_root)
    assert len(out)
    assert all(map(lambda f: ".git" not in str(f), out))
    outf = crawl(project_root, ignore=".vscode")
    assert all(map(lambda f: ".vscode" not in str(f), outf))
    assert len(outf) < len(out)
