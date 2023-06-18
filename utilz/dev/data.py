# %%
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from functools import cached_property
from utilz import randdf
from shutil import rmtree


# %%
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


# %%
def test_subject():
    # Setup data folders
    DATA_DIR = Path(".")
    SID = "sid001"
    SUB_PATH = DATA_DIR / SID
    SUB_PATH.mkdir(exist_ok=True)
    FNAME = "data.csv"

    # Make data
    randdf().to_csv(SUB_PATH / FNAME, index=False)

    # Create subject
    s = Subject(SID, DATA_DIR)

    print(s.data)

    # Second time it's cached
    print(s.data)

    # Remove data
    rmtree(SUB_PATH)


# %%
test_subject()

# %%
