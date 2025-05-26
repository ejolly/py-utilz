"""
Utility functions for polars DataFrame operations.

This module provides helper functions to support dplyr-like verbs for polars DataFrames,
including type checking, expression handling, and compatibility layers.
"""

import polars as pl
from typing import Union, Callable
from functools import wraps

__all__ = [
    "is_polars_dataframe",
    "is_polars_lazyframe", 
    "is_polars_object",
    "ensure_polars_expr",
    "polars_curry",
    "handle_string_expr",
    "convert_pandas_dtype",
    "ensure_pandas_for_plotting",
]

# Type checking utilities
def is_polars_dataframe(obj) -> bool:
    """Check if object is a polars DataFrame."""
    return isinstance(obj, pl.DataFrame)

def is_polars_lazyframe(obj) -> bool:
    """Check if object is a polars LazyFrame.""" 
    return isinstance(obj, pl.LazyFrame)

def is_polars_object(obj) -> bool:
    """Check if object is any polars DataFrame or LazyFrame."""
    return is_polars_dataframe(obj) or is_polars_lazyframe(obj)

# Expression utilities
def ensure_polars_expr(expr: Union[str, pl.Expr, Callable]) -> pl.Expr:
    """
    Convert various input types to polars expressions.
    
    Parameters
    ----------
    expr : str, pl.Expr, or callable
        Expression to convert
        
    Returns
    -------
    pl.Expr
        Polars expression
    """
    if isinstance(expr, pl.Expr):
        return expr
    elif isinstance(expr, str):
        # Handle string expressions using pl.sql_expr for SQL-like syntax
        # or try to parse as column name
        try:
            return pl.sql_expr(expr)
        except:
            # Fall back to treating as column name
            return pl.col(expr)
    elif callable(expr):
        # For lambda functions, we'll need to apply them differently
        # This is a placeholder - actual implementation will depend on context
        raise NotImplementedError("Lambda expressions not yet supported")
    else:
        raise TypeError(f"Cannot convert {type(expr)} to polars expression")

def handle_string_expr(expr_str: str) -> pl.Expr:
    """
    Handle string expressions similar to dplyr/pandas query syntax.
    
    Parameters
    ----------
    expr_str : str
        String expression to parse
        
    Returns  
    -------
    pl.Expr
        Parsed polars expression
    """
    # Simple implementation - can be enhanced for more complex expressions
    return pl.sql_expr(expr_str)

# Curry decorator for polars compatibility
def polars_curry(func: Callable) -> Callable:
    """
    Curry decorator that works with both pandas and polars DataFrames.
    
    This is similar to toolz.curry but includes type checking for polars objects.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if len(args) == 0:
            # Return curried function
            return lambda df: func(df, *args[1:], **kwargs) if len(args) > 1 else lambda df: func(df, **kwargs)
        elif len(args) == 1 and is_polars_object(args[0]):
            # If only DataFrame provided, return curried function
            df = args[0]
            return lambda *rest_args, **rest_kwargs: func(df, *rest_args, **rest_kwargs)
        else:
            # Execute function normally
            return func(*args, **kwargs)
    return wrapper

# Type conversion utilities
def convert_pandas_dtype(pandas_dtype: str) -> pl.DataType:
    """
    Convert pandas dtype string to polars DataType.
    
    Parameters
    ----------
    pandas_dtype : str
        Pandas dtype string
        
    Returns
    -------
    pl.DataType
        Corresponding polars data type
    """
    dtype_mapping = {
        'int64': pl.Int64,
        'int32': pl.Int32, 
        'float64': pl.Float64,
        'float32': pl.Float32,
        'object': pl.Utf8,
        'string': pl.Utf8,
        'bool': pl.Boolean,
        'datetime64[ns]': pl.Datetime,
        'category': pl.Categorical,
    }
    
    return dtype_mapping.get(pandas_dtype, pl.Utf8)  # Default to string


# Plotting compatibility utilities
def ensure_pandas_for_plotting(data):
    """
    Ensure data is a pandas DataFrame for plotting compatibility.
    
    Parameters
    ----------
    data : pl.DataFrame, pd.DataFrame, or other
        Data to potentially convert
        
    Returns
    -------
    pd.DataFrame or original data
        Pandas DataFrame if input was polars, otherwise original data
    """
    if is_polars_object(data):
        # Convert polars to pandas for plotting
        return data.to_pandas()
    else:
        # Return as-is (already pandas or other type)
        return data