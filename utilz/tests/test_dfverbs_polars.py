"""
Test suite for polars DataFrame support in dfverbs module.
"""

import pytest
import polars as pl
import pandas as pd
from utilz import pipe
import utilz.dfverbs as _
import utilz.dfverbs.stats as stats


def test_mutate_polars():
    """Test mutate verb with polars DataFrames."""
    df = pl.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': [10, 20, 30, 40, 50]
    })
    
    # Test string expression
    result = pipe(df, _.mutate(c='a + b'))
    assert 'c' in result.columns
    assert result['c'].to_list() == [11, 22, 33, 44, 55]
    
    # Test literal value
    result = pipe(df, _.mutate(d=100))
    assert 'd' in result.columns
    assert all(result['d'] == 100)
    
    # Test polars expression
    result = pipe(df, _.mutate(e=pl.col('a') * 2))
    assert 'e' in result.columns
    assert result['e'].to_list() == [2, 4, 6, 8, 10]


def test_transmute_polars():
    """Test transmute verb with polars DataFrames."""
    df = pl.DataFrame({
        'a': [1, 2, 3],
        'b': [4, 5, 6],
        'c': [7, 8, 9]
    })
    
    result = pipe(df, _.transmute(sum='a + b', prod='a * b'))
    assert result.shape == (3, 2)
    assert 'sum' in result.columns
    assert 'prod' in result.columns
    assert result['sum'].to_list() == [5, 7, 9]
    assert result['prod'].to_list() == [4, 10, 18]


def test_select_polars():
    """Test select verb with polars DataFrames."""
    df = pl.DataFrame({
        'a': [1, 2, 3],
        'b': [4, 5, 6],
        'c': [7, 8, 9]
    })
    
    # Select specific columns
    result = pipe(df, _.select('a', 'c'))
    assert result.columns == ['a', 'c']
    
    # Drop columns with '-' prefix
    result = pipe(df, _.select('-b'))
    assert result.columns == ['a', 'c']


def test_rename_polars():
    """Test rename verb with polars DataFrames."""
    df = pl.DataFrame({'old_a': [1, 2, 3], 'old_b': [4, 5, 6]})
    
    # Rename with dict
    result = pipe(df, _.rename({'old_a': 'new_a', 'old_b': 'new_b'}))
    assert result.columns == ['new_a', 'new_b']
    
    # Rename with tuple
    result = pipe(df, _.rename(('old_a', 'renamed_a')))
    assert 'renamed_a' in result.columns


def test_query_polars():
    """Test query/filter verb with polars DataFrames."""
    df = pl.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': [10, 20, 30, 40, 50]
    })
    
    # Test string expression
    result = pipe(df, _.query('a > 2'))
    assert result.shape[0] == 3
    assert result['a'].to_list() == [3, 4, 5]
    
    # Test complex expression
    result = pipe(df, _.query('(a > 2) & (b < 50)'))
    assert result.shape[0] == 2
    assert result['a'].to_list() == [3, 4]


def test_groupby_summarize_polars():
    """Test groupby and summarize verbs with polars DataFrames."""
    df = pl.DataFrame({
        'group': ['A', 'A', 'B', 'B', 'C'],
        'value': [10, 20, 30, 40, 50],
        'other': [1, 2, 3, 4, 5]
    })
    
    result = pipe(
        df,
        _.groupby('group'),
        _.summarize(
            mean_val='value.mean()',
            sum_val='value.sum()',
            count='value.count()'
        )
    )
    
    assert result.shape == (3, 4)
    assert 'mean_val' in result.columns
    assert 'sum_val' in result.columns
    assert 'count' in result.columns
    
    # Check specific values
    a_row = result.filter(pl.col('group') == 'A')
    assert a_row['mean_val'][0] == 15.0
    assert a_row['sum_val'][0] == 30
    assert a_row['count'][0] == 2


def test_sort_polars():
    """Test sort verb with polars DataFrames."""
    df = pl.DataFrame({
        'a': [3, 1, 4, 1, 5],
        'b': [30, 10, 40, 20, 50]
    })
    
    # Sort by single column
    result = pipe(df, _.sort('a'))
    assert result['a'].to_list() == [1, 1, 3, 4, 5]
    
    # Sort by multiple columns
    result = pipe(df, _.sort('a', 'b'))
    assert result['a'].to_list() == [1, 1, 3, 4, 5]
    assert result['b'].to_list() == [10, 20, 30, 40, 50]
    
    # Sort descending
    result = pipe(df, _.sort('a', ascending=False))
    assert result['a'].to_list() == [5, 4, 3, 1, 1]


def test_astype_polars():
    """Test astype verb with polars DataFrames."""
    df = pl.DataFrame({
        'a': [1, 2, 3],
        'b': ['10', '20', '30']
    })
    
    # Cast single column
    result = pipe(df, _.astype(('b', 'int64')))
    assert result['b'].dtype == pl.Int64
    assert result['b'].to_list() == [10, 20, 30]
    
    # Cast multiple columns
    df2 = pl.DataFrame({
        'x': [1.0, 2.0, 3.0],
        'y': [1, 0, 1]  # Use numeric values for boolean casting
    })
    result = pipe(df2, _.astype({'x': 'int32', 'y': 'bool'}))
    assert result['x'].dtype == pl.Int32
    assert result['y'].dtype == pl.Boolean


def test_fillna_replace_polars():
    """Test fillna and replace verbs with polars DataFrames."""
    # Test fillna
    df = pl.DataFrame({
        'a': [1, None, 3, None, 5],
        'b': [10, 20, None, 40, None]
    })
    
    result = pipe(df, _.fillna(0))
    assert result['a'].to_list() == [1, 0, 3, 0, 5]
    assert result['b'].to_list() == [10, 20, 0, 40, 0]
    
    # Test replace
    df2 = pl.DataFrame({
        'a': [1, 2, 3],
        'b': ['x', 'y', 'x']
    })
    
    result = pipe(df2, _.replace('x', 'z'))
    assert result['b'].to_list() == ['z', 'y', 'z']


def test_pivot_longer_polars():
    """Test pivot_longer verb with polars DataFrames."""
    df = pl.DataFrame({
        'id': [1, 2, 3],
        'A': [10, 20, 30],
        'B': [40, 50, 60]
    })
    
    result = pipe(
        df,
        _.pivot_longer(['A', 'B'], into=('variable', 'value'))
    )
    
    assert result.shape == (6, 3)
    assert 'variable' in result.columns
    assert 'value' in result.columns
    assert set(result['variable'].unique()) == {'A', 'B'}
    
    # Test with id_vars
    result2 = pipe(
        df,
        _.pivot_longer(id_vars='id', into=('var', 'val'))
    )
    assert result2.shape == (6, 3)
    assert 'var' in result2.columns


def test_pivot_wider_polars():
    """Test pivot_wider verb with polars DataFrames."""
    df = pl.DataFrame({
        'id': [1, 1, 2, 2],
        'variable': ['A', 'B', 'A', 'B'],
        'value': [10, 20, 30, 40]
    })
    
    result = pipe(
        df,
        _.pivot_wider('variable', using='value')
    )
    
    assert result.shape == (2, 3)
    assert 'A' in result.columns
    assert 'B' in result.columns
    assert result.filter(pl.col('id') == 1)['A'][0] == 10
    assert result.filter(pl.col('id') == 1)['B'][0] == 20


def test_concat_merge_join_polars():
    """Test concat, merge, and join verbs with polars DataFrames."""
    # Test concat
    df1 = pl.DataFrame({'a': [1, 2], 'b': [3, 4]})
    df2 = pl.DataFrame({'a': [5, 6], 'b': [7, 8]})
    
    result = _.concat([df1, df2])
    assert result.shape == (4, 2)
    assert result['a'].to_list() == [1, 2, 5, 6]
    
    # Test merge
    df3 = pl.DataFrame({'a': [1, 2], 'c': [10, 20]})
    result = _.merge(df1, df3, on='a')
    assert result.shape == (2, 3)
    assert 'c' in result.columns
    
    # Test join
    result = pipe(df1, _.join(df3, on='a'))
    assert result.shape == (2, 3)
    assert 'c' in result.columns


def test_statistical_functions_polars():
    """Test statistical functions with polars DataFrames."""
    df = pl.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': [10, 20, 30, 40, 50]
    })
    
    # Test mean
    result = pipe(df, stats.mean())
    assert isinstance(result, pl.DataFrame)
    assert result['a'][0] == 3.0
    assert result['b'][0] == 30.0
    
    # Test std
    result = pipe(df, stats.std())
    assert isinstance(result, pl.DataFrame)
    assert result['a'][0] > 1.5  # Should be ~1.58
    
    # Test min/max
    result_min = pipe(df, stats.min())
    result_max = pipe(df, stats.max())
    assert result_min['a'][0] == 1
    assert result_max['a'][0] == 5
    
    # Test sum
    result = pipe(df, stats.sum())
    assert result['a'][0] == 15
    assert result['b'][0] == 150
    
    # Test count
    result = pipe(df, stats.count())
    assert result['a'][0] == 5
    assert result['b'][0] == 5


def test_split_polars():
    """Test split verb with polars DataFrames."""
    # Test string splitting
    df = pl.DataFrame({
        'name': ['John-Doe', 'Jane-Smith', 'Bob-Jones'],
        'age': [25, 30, 35]
    })
    
    result = pipe(df, _.split('name', ['first', 'last'], sep='-'))
    assert 'first' in result.columns
    assert 'last' in result.columns
    assert 'name' not in result.columns
    assert result['first'].to_list() == ['John', 'Jane', 'Bob']
    assert result['last'].to_list() == ['Doe', 'Smith', 'Jones']
    
    # Test list splitting
    df2 = pl.DataFrame({
        'values': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        'id': ['a', 'b', 'c']
    })
    
    result2 = pipe(df2, _.split('values', ['v1', 'v2', 'v3'], sep=[]))
    assert 'v1' in result2.columns
    assert 'v2' in result2.columns
    assert 'v3' in result2.columns
    assert result2['v1'].to_list() == [1, 4, 7]


def test_head_tail_polars():
    """Test head and tail verbs with polars DataFrames."""
    df = pl.DataFrame({
        'a': list(range(1, 11)),
        'b': list(range(10, 20))
    })
    
    # Test head
    result = pipe(df, _.head(3))
    assert result.shape == (3, 2)
    assert result['a'].to_list() == [1, 2, 3]
    
    # Test tail
    result = pipe(df, _.tail(3))
    assert result.shape == (3, 2)
    assert result['a'].to_list() == [8, 9, 10]


def test_pandas_polars_consistency():
    """Test that operations produce consistent results between pandas and polars."""
    # Create identical dataframes
    data = {'a': [1, 2, 3], 'b': [4, 5, 6]}
    df_pd = pd.DataFrame(data)
    df_pl = pl.DataFrame(data)
    
    # Test mutate
    result_pd = pipe(df_pd, _.mutate(c='a + b'))
    result_pl = pipe(df_pl, _.mutate(c='a + b'))
    assert result_pd['c'].tolist() == result_pl['c'].to_list()
    
    # Test select
    result_pd = pipe(df_pd, _.select('a'))
    result_pl = pipe(df_pl, _.select('a'))
    assert list(result_pd.columns) == result_pl.columns
    
    # Test rename
    result_pd = pipe(df_pd, _.rename({'a': 'x'}))
    result_pl = pipe(df_pl, _.rename({'a': 'x'}))
    assert list(result_pd.columns) == result_pl.columns


def test_plotting_with_polars():
    """Test that plotting functions work with polars DataFrames."""
    import utilz.dfverbs as _
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for testing
    import matplotlib.pyplot as plt
    
    # Create test data
    df = pl.DataFrame({
        'x': [1, 2, 3, 4, 5],
        'y': [2, 4, 6, 8, 10],
        'group': ['A', 'A', 'B', 'B', 'A']
    })
    
    # Test scatterplot
    try:
        plot = pipe(df, _.scatterplot(x='x', y='y'))
        assert plot is not None
        plt.close('all')
    except Exception as e:
        pytest.fail(f"Scatterplot failed with polars DataFrame: {e}")
    
    # Test barplot  
    try:
        plot = pipe(df, _.barplot(x='group', y='y'))
        assert plot is not None
        plt.close('all')
    except Exception as e:
        pytest.fail(f"Barplot failed with polars DataFrame: {e}")
    
    # Test that conversion happens
    from utilz.dfverbs.polars_utils import ensure_pandas_for_plotting
    converted = ensure_pandas_for_plotting(df)
    assert isinstance(converted, pd.DataFrame)
    assert converted.shape == df.shape


def test_read_csv_polars():
    """Test read_csv with use_polars parameter."""
    # Create a temporary CSV file
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write('a,b,c\n1,2,3\n4,5,6\n')
        temp_path = f.name
    
    try:
        # Read with polars
        df_pl = _.read_csv(temp_path, use_polars=True)
        assert isinstance(df_pl, pl.DataFrame)
        assert df_pl.shape == (2, 3)
        
        # Read with pandas (default)
        df_pd = _.read_csv(temp_path)
        assert isinstance(df_pd, pd.DataFrame)
        assert df_pd.shape == (2, 3)
    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])