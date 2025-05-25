# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current Refactoring Goals for 0.5.0 Release

These are the broad refactoring goals of the current git branch `uv-polars`:

- [x] Switch from `poetry` to `uv` for dependency management
- [ ] Simplify maps and pipes to remove superfluous functionality (e.g. retain `map` but not `mapcat`) and simplify over-engineering
- [ ] Refactor to remove `pandas` and switch to `polars`, while leveraging strengths of `polars` (e.g custom expressions?) to achieve feature parity with the `dplyr` library in R as demonstrated in this tutorial: https://jhudatascience.org/tidyversecourse/wrangle-data.html#data-wrangling

For more details you should refer to [REFACTOR.md](./REFACTOR.md) which acts as a live-log of the plan's progress.

## Coding Rules to Follow

- When you are unsure about an implementation detail and cannot determine it yourself, check the relevant library's API documentation using your WebFetch tool
- If you are ever uncertain about how to proceed and cannot figure something out yourself, you should stop and ask the user for feedback
- Make sure to adhere to coding conventions as expected by `ruff`
- Use google-style docstrings
- Use modern Python type annotations
- Make to sure to use `uv` to run commands and manage the project
- Prefer running tests one-at-a-time during development to iterate quickly. Then run the entire test suite when you're done implementing to ensure nothing else was negatively affected by your changes.
- Never "hack tests" just to make them pass. They should always test genuine intended functionality
- Never git commit if tests are broken
- If there exists a TODOS.md file, you should read it and use that to seed your current todo list if you dont already have one
- At committing your work, you should also take a snapshot of your current todo list to TODOS.md, creating one if it doesn't exist, or updating one if it does
- Always re-read this CLAUDE.md file after writing to TODOS.md. Verify you have done this by printing the statement: "I'VE REMINDED MYSELF OF THE OBJECTIVES OF MY WORK"

## Project Overview

py-utilz is a utility library providing functional programming tools, data analysis utilities, and convenience functions with minimal dependencies. It features dplyr-like data grammar, pipes, decorators, and various helper functions.

## Essential Commands

```bash
# Run tests
uv run pytest

# Run specific test file
uv run pytest utilz/tests/test_pipes.py

# Run specific test
uv run pytest utilz/tests/test_pipes.py -k 'test_function_name'

# Lint code
uv run ruff check

# Build documentation locally
uv run mkdocs serve  # Live preview at http://127.0.0.1:8000
uv run mkdocs build  # Build static site
```

## Architecture & Conventions

### Module Structure

- Each module focuses on specific functionality (pipes, decorators, data tools, etc.)
- Modules use explicit `__all__` exports and are re-exported through `__init__.py`
- Heavy use of functional programming patterns: currying, composition, and pipes

### Key Modules

- `pipes.py`: Pipe operator and functional composition tools
- `decorators.py`: Logging, caching, and defensive programming decorators
- `dfverbs/`: dplyr-like verbs for dataframe manipulation
- `data.py`: Data generation and manipulation utilities
- `plot.py`: Plotting utilities built on matplotlib/seaborn

### Testing Approach

- Tests mirror source structure in `utilz/tests/`
- Use simple assertion-based testing
- Test both normal operation and edge cases
- Common test patterns include testing curried functions in pipes
