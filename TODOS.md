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

## Pending Tasks - Pandas to Polars Migration

### Phase 4: dfverbs Module Migration

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
  - [ ] Port all statistical functions from stats.py
  - [ ] Implement grouped operations patterns for mutate
  - [ ] Add lazy evaluation support

- [ ] Step 3: Reshaping and Joining Operations
  - Migrate pivot_longer/pivot_wider to melt/pivot
  - Implement split verb with polars
  - Update all joining operations
  - Port sorting and selection verbs

- [ ] Step 4: Plotting and Integration
  - Create polars→pandas conversion layer for plotting
  - Update all plot functions to handle polars
  - Implement I/O operations (read_csv, to_csv)
  - Port utility functions (apply, call, type conversions)

- [ ] Step 5: Testing and Documentation
  - Migrate all dfverbs tests to polars
  - Update documentation and notebooks
  - Create user migration guide
  - Benchmark performance improvements

## Notes
- All tests pass after each step
- Documentation is updated with migration examples
- Following the plan in REFACTOR.md
- Polars migration aims to maintain API compatibility while leveraging performance benefits