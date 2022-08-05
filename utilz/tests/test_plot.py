from utilz.plot import mpinit, stripbarplot, savefig
import matplotlib.pyplot as plt
from pathlib import Path


def test_mpinit():
    f, axs = mpinit(subplots=(2, 2))
    assert f is not None
    assert axs.shape == (2, 2)
    plt.close(f)


def test_stripbarplot():
    pass


def test_savefig(tmp_path: Path):

    # Save to cwd
    f, _ = mpinit(subplots=(2, 2))
    save_raster = Path.cwd() / "test.jpg"
    save_vector = Path.cwd() / "test.pdf"
    savefig(f, "test")
    plt.close(f)
    assert save_raster.exists()
    assert save_vector.exists()
    save_raster.unlink()
    save_vector.unlink()

    # Save to custom path
    tmp_path = Path(tmp_path)
    dir_save_raster = tmp_path / "test.jpg"
    dir_save_vector = tmp_path / "test.pdf"

    f, _ = mpinit(subplots=(2, 2))
    savefig(f, "test", path=tmp_path)
    plt.close(f)
    assert dir_save_raster.exists()
    assert dir_save_vector.exists()

    dir_save_raster.unlink()
    dir_save_vector.unlink()
