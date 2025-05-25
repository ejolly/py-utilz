# Refactoring Plan

## Overview

This document outlines the refactoring plans for py-utilz 0.5.0 release, including:
1. Simplifying the `utilz.maps` and `utilz.pipes` modules (partially complete)
2. Migrating `utilz.dfverbs` from pandas to polars while maintaining dplyr-like functionality

## Current State Analysis

### Maps Module Issues
- **Too many specialized variants**: 7 different map functions (map, mapcat, mapcompose, mapmany, mapacross, mapif, mapwith)
- **Complex implementation**: Mixing parallel execution logic with functional transformations
- **Redundant functionality**: Many functions duplicate what can be achieved by composing simpler primitives
- **Over-engineered features**: Automatic concatenation, type detection, and special handling

### Pipes Module Issues  
- **Complex flags**: pipe() has too many flags (debug, save, load_existing, flatten, keep, etc.)
- **Overlapping functions**: alongwith, append, and spread have similar purposes
- **Magical behavior**: Ellipsis (...) handling adds unnecessary complexity
- **Dual-purpose functions**: spread tries to be both fork and juxt

## Target Architecture

### Maps Module - Simplified API
**Keep only:**
- `map` - Core mapping function with parallel support
- `filter` - Current implementation is good
- `check_random_state` - Utility function

**Remove:**
- `mapcat` → Use `pipe(items, map(func), pd.concat)`
- `mapcompose` → Use `map(compose(f2, f1), items)`  
- `mapmany` → Use `map(juxt(f1, f2), items)`
- `mapacross` → Too specialized, confusing semantics
- `mapif` → Use `map(iffy(predicate, func), items)`
- `mapwith` → Over-engineered, use `zip` + `map`

### Pipes Module - Simplified API
**Keep only:**
- `pipe` - Core function, simplified to only have `output` and `show` flags
- `fork` - Simple duplication utility
- `gather`/`unpack` - Useful for unpacking tuples

**Remove:**
- `alongwith`/`append` → Use `lambda x: (x, func(x))`
- `spread` → Use `fork` + `map` or `juxt`
- `pop` → Too specialized, use standard tuple operations

## Refactoring Steps

Each step follows this pattern:
1. Make the code change
2. Update/mark tests 
3. Update documentation with migration examples
4. Run tests and lint
5. Commit changes

### Phase 1: Maps Module

#### Step 1: Remove mapcat
- Remove function from maps.py
- Update __all__ export
- Mark/update tests in test_ops.py
- Add migration example to docstring before removal
- Migration: `mapcat(func, items)` → `pipe(items, map(func), pd.concat)`

#### Step 2: Remove mapcompose  
- Remove function from maps.py
- Update __all__ export
- Mark/update tests
- Add migration example to docstring
- Migration: `mapcompose(f1, f2, items)` → `map(compose(f2, f1), items)`

#### ~~Step 3: Remove mapmany~~ ✓
- Completely removed mapmany function and _many helper from maps.py
- Updated __all__ exports to exclude mapmany
- Updated module docstring to remove mapmany reference
- Updated test cases to use map(juxt(...)) pattern instead
- Updated error message in ops.py to suggest juxt pattern
- Added migration documentation in pipes.md
- All tests pass (30 passed, 18 skipped)

#### ~~Step 4: Remove mapacross~~ ✓
- Completely removed mapacross function from maps.py
- Updated __all__ exports to exclude mapacross
- Updated module docstring to remove mapacross reference
- Updated test cases to use explicit zip + list comprehension pattern
- Updated documentation in pipes.md with migration examples
- All tests pass (30 passed, 18 skipped)

#### ~~Step 5: Remove mapif~~ ✓
- Completely removed mapif function from maps.py
- Updated __all__ exports to exclude mapif
- Updated module docstring to remove mapif reference
- Removed unused iffy import from maps.py
- Updated test cases to use map(iffy(...)) pattern
- Added migration documentation in pipes.md
- All tests pass (30 passed, 18 skipped)

#### ~~Step 6: Remove mapwith~~ ✓
- Completely removed mapwith function from maps.py
- Updated __all__ exports to exclude mapwith
- Updated module docstring to remove mapwith reference
- Removed unused deepcopy import from maps.py
- Updated test cases to use standard Python patterns (closures and zip)
- Added migration documentation in pipes.md
- All tests pass (30 passed, 18 skipped)

#### Step 7: Simplify map()
- Remove `enum` parameter (use enumerate())
- Remove concatenation logic
- Simplify kwargs handling
- Keep only essential parameters: func, iterme, n_jobs, backend, pbar, verbose, random_state

### Phase 2: Pipes Module

#### Step 8: Remove alongwith/append
- Remove both functions from pipes.py
- Update __all__ export
- Mark/update tests
- Add migration example to docstring
- Migration: `alongwith(func)` → `lambda x: (x, func(x))`

#### Step 9: Remove spread
- Remove function from pipes.py  
- Update __all__ export
- Mark/update tests
- Add migration example to docstring
- Migration: Use `fork` + `map` or direct `juxt` usage

#### Step 10: Remove pop
- Remove function from pipes.py
- Update __all__ export
- Mark/update tests
- Add migration example to docstring
- Migration: Use standard tuple slicing

#### Step 11: Simplify pipe()
- Remove flags: debug, keep, load_existing, save, flatten
- Remove ellipsis (...) support
- Keep only: output, show flags
- Simplify implementation

### Phase 3: Finalization

#### Step 12: Update module exports
- Update utilz/__init__.py to reflect new exports
- Ensure backward compatibility imports still work with deprecation warnings

#### Step 13: Final testing
- Run full test suite
- Ensure all tests pass or are properly marked
- Run linting

## Migration Guide

### Common Patterns

```python
# Old: mapcat for concatenating results
result = mapcat(lambda x: [x, x*2], [1, 2, 3])

# New: explicit map + concatenation  
result = pipe([1, 2, 3], map(lambda x: [x, x*2]), list, lambda x: sum(x, []))

# Old: mapcompose for sequential operations
result = mapcompose(add_1, multiply_2, [1, 2, 3])

# New: compose functions first
result = map(compose(multiply_2, add_1), [1, 2, 3])

# Old: alongwith to keep previous values
result = pipe(data, alongwith(process))

# New: explicit tuple creation
result = pipe(data, lambda x: (x, process(x)))

# Old: spread for multiple operations
result = pipe(data, spread(filter1, filter2, filter3))

# New: use juxt or explicit fork + map
result = pipe(data, juxt(filter1, filter2, filter3))
```

## Testing Strategy

1. **Deprecation warnings**: Add warnings to functions before removal
2. **Test marking**: Mark tests for deprecated functions with pytest.mark.skip or update to use new patterns
3. **New tests**: Add tests for migration patterns to ensure they work correctly
4. **Documentation**: Update all docstrings with clear migration examples

## Benefits

1. **Reduced API surface**: From 13 functions to 6 core functions
2. **Better composability**: Users combine simple primitives  
3. **Less magic**: Explicit behavior, no hidden side effects
4. **Cleaner code**: Easier to maintain and understand
5. **Better separation**: Parallel execution separate from functional transformations

## Current Progress

Track progress using the TodoRead/TodoWrite tools. Each numbered step above corresponds to a set of todo items that should be completed before moving to the next step.

### Completed Steps

#### ~~Step 1: Remove mapcat~~ ✓
- Completely removed mapcat function and _concat helper from maps.py
- Updated usages in io.py and dftools.py to use pipe + map + list
- Removed mapcat test from test_ops.py
- Updated documentation in pipes.md
- Fixed unused pandas import
- All tests pass (11 passed in test_ops.py)

#### ~~Step 2: Remove mapcompose~~ ✓
- Completely removed mapcompose function from maps.py
- Updated __all__ exports to exclude mapcompose
- Updated module docstring to remove mapcompose reference
- Removed unused compose import from maps.py
- Updated test cases to use map(compose(...)) pattern
- Added migration documentation in pipes.md with correct compose_left order
- All tests pass (11 passed in test_ops.py)

## Pandas to Polars Migration Plan for dfverbs

### Overview

Migrate `utilz.dfverbs` from pandas to polars while maintaining dplyr-like functionality and leveraging polars' performance benefits.

### Migration Steps

#### Step 1: Infrastructure and Core Verbs Migration
**Components:**
- Add polars dependency to pyproject.toml
- Create `utilz/dfverbs/polars_utils.py` with helper functions for polars operations
- Update curry decorators to work with polars DataFrames and LazyFrames
- Implement type checking utilities for polars objects

**Core verb migrations:**
- `mutate` → `with_columns()` with polars expressions
- `transmute` → `select()` with new column expressions
- `select` → `select()` with enhanced column selection patterns
- `filter`/`query` → `filter()` with polars expressions (support string expressions via pl.sql_expr)
- `rename` → `rename()` mapping
- `astype` → `cast()` with polars dtypes
- `fillna`/`replace` → `fill_null()`/`replace()`

**Expression support:**
- Enable string expressions using `pl.sql_expr()` or custom parser
- Support lambda functions with automatic column detection
- Create expression builders for common patterns

#### Step 2: Aggregation and Grouping Operations
**Grouping migrations:**
- `groupby` → `group_by()` with support for maintaining grouped state
- `summarize` → `group_by().agg()` with polars aggregation expressions
- `ngroups`, `get_group`, `split_groups` → Polars equivalents
- Implement grouped operations pattern for other verbs

**Statistical functions (stats.py):**
- Basic stats: `mean`, `median`, `min`, `max`, `sum`, `std`, `var`, `count`
- Advanced stats: `mode` (custom implementation), `sem`, `quantile`
- Special functions: `bootci` (implement using polars sampling), `corr`, `cov`
- Aggregation helpers: `nunique`, `unique`, `value_counts`, `rank`, `size`
- Math operations: `round`, `abs`, `sqrt`, `all`, `any`

#### Step 3: Reshaping and Joining Operations
**Reshaping verbs:**
- `pivot_longer` → `melt()`/`unpivot()` with column selection
- `pivot_wider` → `pivot()` with value aggregation options
- `split` → Custom implementation using `str.split()` and `with_columns()`

**Joining and combining:**
- `merge`/`join` → `join()` with various join strategies
- `concat` → `concat()` for combining DataFrames
- Add support for `join_asof()` for time-based joins

**Ordering and selection:**
- `sort`/`arrange` → `sort()` with multiple columns
- `head`/`tail` → Native polars methods
- `slice` → `slice()` or filter with row indices

#### Step 4: Plotting and Integration
**Plotting compatibility:**
- Create `to_pandas()` wrapper for automatic conversion before plotting
- Implement caching strategy for repeated conversions
- Update all plot functions in plot.py to handle polars DataFrames
- Maintain existing seaborn integration

**I/O operations:**
- `read_csv` → `pl.read_csv()` with similar parameters
- `to_csv` → `write_csv()` method
- Add support for other formats (parquet, json) if needed

**Utility functions:**
- `apply` → Use polars expressions or `map_elements()` for custom functions
- `call` → Generic method caller for polars DataFrames
- Type conversions: `squeeze`, `to_numpy`, `to_list` → Polars equivalents

#### Step 5: Testing and Documentation
**Testing updates:**
- Create polars-specific test fixtures
- Update all tests in test_dfverbs.py for polars
- Add comparison tests to ensure pandas/polars parity where expected
- Performance benchmarks comparing pandas vs polars implementations

**Documentation updates:**
- Update all docstrings with polars examples
- Revise notebooks: verbs.ipynb, intro.ipynb
- Create migration guide showing pandas → polars verb mappings
- Update api/df.md with polars-specific notes

**Additional features:**
- Add `lazy` parameter to verbs for lazy evaluation support
- Implement `collect` verb for materializing lazy frames
- Create compatibility layer for gradual migration (optional)

### Key Polars Advantages to Leverage

1. **Expression API**: Use polars expressions for complex transformations
2. **Lazy Evaluation**: Enable query optimization and reduced memory usage
3. **Parallel Operations**: Automatic parallelization of operations
4. **Memory Efficiency**: Columnar storage and zero-copy operations
5. **Type Safety**: Stronger type system than pandas

### Migration Priority

1. Core verbs (highest usage): mutate, select, filter, summarize, groupby
2. Reshaping operations: pivot_longer, pivot_wider
3. Statistical functions
4. Plotting integration
5. Advanced features and optimizations