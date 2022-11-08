"""Quick common stats functions"""

__all__ = ["scale"]

import numpy as np
from numpy.typing import ArrayLike
from toolz import curry


@curry
def scale(
    arr: ArrayLike, center: bool = True, scale: bool = True, axis: int = 0
) -> ArrayLike:
    """
    R like scale function for centering, norming, or zscoring an arraylike object

    Args:
        arr (ArrayLike): array like object
        center (bool, optional): whether to subtract the mean. Defaults to True.
        scale (bool, optional): whether to divide by the std. Defaults to True.
        axis (int, optional): axis to compute mean and std over if arr >= 2d. Defaults to 0.

    Returns:
        ArrayLike: same shape as input
    """

    if center and scale:
        return (arr - np.mean(arr, axis=axis)) / np.std(arr, axis=axis)
    if center:
        return arr - np.mean(arr, axis=axis)
    if scale:
        return arr / np.std(arr, axis=axis)
