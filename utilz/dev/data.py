# %%
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from functools import cached_property
from utilz import randdf, equal
from shutil import rmtree
from utilz import discard


# %% Subject data container
@dataclass
class Subject:
    """Data container that makes uses of new cached_property decorator"""

    sid: str
    data_dir: Path

    def __post_init__(self):
        self.data_dir = self.data_dir / self.sid

    @cached_property
    def data(self):
        print("Loading data")
        return pd.read_csv(self.data_dir / "data.csv")

    def clear_cache(self):
        del self.data


# %% Subject data test
def test_subject():
    # Setup data folders
    DATA_DIR = Path(".")
    SID = "sid001"
    SUB_PATH = DATA_DIR / SID
    SUB_PATH.mkdir(exist_ok=True)
    FNAME = "data.csv"

    # Make data
    data = randdf()
    data.to_csv(SUB_PATH / FNAME, index=False)

    # Create subject
    s = Subject(SID, DATA_DIR)

    assert equal(s.data, data)

    # Change data on disk
    data = randdf()
    data.to_csv(SUB_PATH / FNAME, index=False)

    # Second time it's cached
    assert not equal(s.data, data)

    # Clear cache
    s.clear_cache()

    # Third time it's not cached
    assert equal(s.data, data)

    # Remove data
    rmtree(SUB_PATH)


# %% Box data container


class Box(list):
    def __init__(self, iterable):
        super().__init__(iterable)

    def __getattr__(self, name, *args, **kwargs):
        if hasattr(self[0], name):
            attr_or_method = getattr(self[0], name)
            if callable(attr_or_method):

                def fn(*args, **kwargs):
                    out = []
                    for elem in self:
                        result = getattr(elem, name)
                        result = result(*args, **kwargs) if callable(result) else result
                        out.append(result)
                    return out

                return fn
            else:
                return [getattr(elem, name) for elem in self]
        else:
            raise AttributeError

    def __repr__(self):
        return f"Box(len={len(self)}, type={self[0].__class__.__module__}.{self[0].__class__.__name__})"


# %%
# Mock class that has attribute access to its underlying data
df = randdf()
df.data = df.to_numpy()
b = Box([df, df])
b


# %%
b.data
b.head()

# %%
box = Box([np.random.randn(10) for i in range(10)])

# %% Run Tests
test_subject()

# %%
