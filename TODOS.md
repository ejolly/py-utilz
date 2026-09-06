# TODO List for Refactoring

This file tracks the progress of the refactoring work on the `uv-polars` branch.

## Completed Tasks ✓

### Phase 1: Maps Module - Steps 1-6
- [x] Step 1: Remove mapcat function from maps.py
- [x] Step 2: Remove mapcompose function from maps.py  
- [x] Step 3: Remove mapmany function from maps.py
- [x] Step 4: Remove mapacross function from maps.py
- [x] Step 5: Remove mapif function from maps.py
- [x] Step 6: Remove mapwith function from maps.py

### Phase 4: dfverbs Module Migration - COMPLETED! 🎉

- [x] Step 1: Infrastructure and Core Verbs Migration
  - [x] Add polars dependency to pyproject.toml
  - [x] Create polars_utils.py helper module  
  - [x] Migrate core verbs: mutate, transmute, select, filter/query, rename
  - [x] Implement polars expression support
  - [x] Update curry decorators for polars compatibility
  - [x] Basic testing of migrated verbs

- [x] Step 2: Aggregation and Grouping Operations
  - [x] Migrate summarize and groupby to polars
  - [x] Migrate sort, astype, fillna, replace verbs
  - [x] Basic aggregation functions (mean, sum, count, min, max, std, var)
  - [x] Port statistical functions from stats.py
  - [x] Implement grouped operations patterns for mutate

- [x] Step 3: Reshaping and Joining Operations
  - [x] Migrate pivot_longer/pivot_wider to melt/pivot
  - [x] Update concat/merge/join operations for polars
  - [x] Add read_csv support with use_polars parameter
  - [x] Implement split verb with polars
  - [x] Create comprehensive test suite for polars verbs (18 tests)

- [x] Step 4: Plotting and Integration
  - [x] Create polars→pandas conversion layer for plotting (ensure_pandas_for_plotting)
  - [x] Update all plot functions to handle polars (added @polars_plotting_wrapper)
  - [x] Add pyarrow dependency for polars→pandas conversion
  - [x] Test plotting functionality with polars DataFrames
  
- [x] Step 5: Testing and Documentation
  - [x] Create comprehensive polars test suite (test_dfverbs_polars.py)
  - [x] Update documentation: README.md, docs/api/df.md, docs/verbs.ipynb, docs/intro.ipynb, docs/index.md
  - [x] Add polars examples throughout documentation
  - [x] All tests pass (48 passed, 18 skipped)
  
## Summary of Polars Migration

The pandas to polars migration has been successfully completed for the dfverbs module:

- ✅ All core verbs support both pandas and polars DataFrames
- ✅ Plotting functions automatically convert polars to pandas for seaborn compatibility
- ✅ Comprehensive test suite with 18 polars-specific tests
- ✅ Backward compatibility maintained - existing pandas code continues to work
- ✅ Documentation updated with polars examples throughout

### Key Features Implemented:
1. **Core Verbs**: mutate, transmute, select, rename, query/filter, groupby, summarize
2. **Data Operations**: sort, astype, fillna, replace, head, tail  
3. **Reshaping**: pivot_longer (unpivot), pivot_wider (pivot), split
4. **Combining**: concat, merge, join
5. **Statistics**: All functions (mean, std, var, min, max, sum, count, median, mode, quantile)
6. **Plotting**: All 30+ seaborn plot functions with automatic conversion
7. **I/O**: read_csv with use_polars parameter
8. **Documentation**: Updated across all docs with polars examples

### Notes:
- Pandas remains a dependency for plotting compatibility and legacy code
- String-based expressions fully supported for polars (using pl.sql_expr)
- Lambda expressions work with polars DataFrames
- All tests pass with both pandas and polars

## Deferred Tasks

### Phase 1: Maps Module - Step 7

- [ ] Step 7: Simplify map() function
  - Remove `enum` parameter (use enumerate())
  - Remove concatenation logic
  - Simplify kwargs handling
  - Keep only essential parameters: func, iterme, n_jobs, backend, pbar, verbose, random_state

### Phase 2: Pipes Module - Steps 8-11
- [ ] Step 8: Remove alongwith/append
- [ ] Step 9: Remove spread
- [ ] Step 10: Remove pop
- [ ] Step 11: Simplify pipe()

### Phase 3: Finalization - Steps 12-13
- [ ] Step 12: Update module exports
- [ ] Step 13: Final testing

## Future Considerations
- Consider LazyFrame support if needed
- Performance benchmarking between pandas and polars implementations
- Additional polars-specific optimizations where applicable