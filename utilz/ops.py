"""
Common data operations and transformations. Often on pandas dataframes
"""

from cytoolz import memoize, curry

@curry
@memoize
def norm_by_group(df, grpcols, valcol, center=True, scale=True):
    """
    Normalize values in a column separately per group

    Args:
        df (pd.DataFrame): input dataframe
        grpcols (str/list): grouping col(s)
        valcol (str): value col
        center (bool, optional): mean center. Defaults to True.
        scale (bool, optional): divide by standard deviation. Defaults to True.
    """

    def _norm(dat, center, scale):
        if center:
            dat = dat - dat.mean()
        if scale:
            dat = dat / dat.std()
        return dat

    return df.groupby(grpcols)[valcol].transform(_norm, center, scale)

