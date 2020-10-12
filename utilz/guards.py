"""
Custom guard decorators compatible with bulwark
"""

# Convert from: https://github.com/ejolly/engarde
# @same_size(group_col)
# @same_nunique(col)
# @log - print size after each function call

from functools import wraps
import datetime as dt


def log_df(func):
    """
    Log the shape and run time of a function that operates on a pandas dataframe

    Args:
        func (callable): a function that operates on a dataframe

    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        tic = dt.datetime.now()
        result = func(*args, **kwargs)
        time_taken = str(dt.datetime.now() - tic)
        print(f"Func {func.__name__} df shape={result.shape} took {time_taken}s")
        return result

    return wrapper
