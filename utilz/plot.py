"""
Plotting convenience functions
"""

__all__ = ["setup"]


def setup(figsize=(8, 6), subplots=(1, 1)):
    import matplotlib.pyplot as plt

    f, ax = plt.subplots(*subplots, figsize=figsize)
    return f, ax
