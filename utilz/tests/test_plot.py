from utilz.plot import mpinit, stripbarplot, savefig
from utilz.boilerplate import randdf
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def test_mpinit():
    f, axs = mpinit(subplots=(2, 2))
    assert f is not None
    assert axs.shape == (2, 2)
    plt.close(f)


def test_stripbarplot():
    df = randdf((12, 3))
    df["group"] = ["a"] * 4 + ["b"] * 4 + ["c"] * 4
    ax = stripbarplot(x="group", y="A1", data=df)
    assert ax is not None
    plt.close("all")
    f, axs = mpinit()
    out = stripbarplot(x="group", y="A1", data=df, ax=axs, estimator=np.median)
    assert out == axs
    plt.close(f)


def test_savefig(tmp_path: Path):
    # Save to cwd
    f, _ = mpinit(subplots=(2, 2))
    save_raster = Path.cwd() / "raster" / "test.jpg"
    save_vector = Path.cwd() / "vector" / "test.pdf"
    savefig(f, "test")
    plt.close(f)
    assert save_raster.exists()
    assert save_vector.exists()
    save_raster.unlink()
    save_raster.parent.rmdir()
    save_vector.unlink()
    save_vector.parent.rmdir()

    # Save to custom path
    tmp_path = Path(tmp_path)
    dir_save_raster = tmp_path / "test.jpg"
    dir_save_vector = tmp_path / "test.pdf"

    f, _ = mpinit(subplots=(2, 2))
    savefig(f, "test", path=tmp_path, use_subdirs=False)
    plt.close(f)
    assert dir_save_raster.exists()
    assert dir_save_vector.exists()

    dir_save_raster.unlink()
    dir_save_vector.unlink()
