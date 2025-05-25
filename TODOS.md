# TODO List for Refactoring

This file tracks the progress of the refactoring work on the `uv-polars` branch.

## Completed Tasks ✓

### Phase 1: Maps Module - Steps 1-3
- [x] Step 1: Remove mapcat function from maps.py
- [x] Step 2: Remove mapcompose function from maps.py  
- [x] Step 3: Remove mapmany function from maps.py

## Pending Tasks

### Phase 1: Maps Module - Steps 4-7
- [ ] Step 4: Remove mapacross function from maps.py
  - Remove function from maps.py
  - Update __all__ exports to exclude mapacross
  - Update module docstring to remove mapacross reference
  - Update test_ops.py to remove or modify mapacross tests
  - Update documentation with mapacross migration example
  - Commit changes

- [ ] Step 5: Remove mapif function from maps.py
  - Remove function from maps.py
  - Update __all__ exports
  - Update tests
  - Add migration example to documentation
  - Migration: `mapif(func, pred, items)` → `map(iffy(pred, func), items)`

- [ ] Step 6: Remove mapwith function from maps.py
  - Remove function from maps.py
  - Update __all__ exports
  - Update tests
  - Add migration example to documentation
  - Migration: Use standard zip + map pattern

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

## Notes
- All tests pass after each step
- Documentation is updated with migration examples
- Following the plan in REFACTOR.md