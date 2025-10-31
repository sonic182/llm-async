# Agent Instructions for llm_async

## Commands
- **Test all**: `poetry run pytest -p no:sugar`
- **Test single**: `poetry run pytest -k "test_name"`
- **Lint**: `poetry run ruff check llm_async tests`
- **Format**: `poetry run ruff format llm_async tests`
- **Build**: `poetry build`
- **Install deps**: `poetry install`
- **Running examples**: `poetry run python <file.py>`

Note: try to use `-p no:sugar` whenever running tests

## Project Structure
The main package is `llm_async`, managed with Poetry (see `pyproject.toml` for configuration).

**Notes:**
- Google provider uses camelCase for API keys (e.g., `generationConfig`, `responseMimeType`, `responseSchema`) to match Google's API specification.

## Code Style
- **Python**: 3.9+ with Poetry, line length 99 chars
- **Async-first**: Use asyncio for all I/O operations
- **Imports**: stdlib → third-party → local (alphabetical within groups)
- **Types**: Full type hints required, strict mypy settings, use `type: ignore` comments when needed
- **Naming**: PascalCase classes, snake_case methods/functions
- **Error handling**: Raise exceptions for API errors, NotImplementedError for unimplemented features
- **Linting**: ruff with the following rules enabled:
  - **E** (pycodestyle errors) - Python style guide violations
  - **W** (pycodestyle warnings) - Minor style issues
  - **F** (pyFlakes) - Logic errors and unused imports/variables
  - **I** (isort) - Import sorting
  - **B** (flake8-bugbear) - Potential bugs and design issues
  - **C4** (flake8-comprehensions) - Better comprehension practices
  - **UP** (pyupgrade) - Modern Python syntax upgrades
- **Testing**: pytest-asyncio with AsyncMock for HTTP client mocking

## Git Command Safety
- **Do not run any git commands that modify the repository without explicit user approval.**
- Avoid the following commands unless the user explicitly requests and approves them:
  - `git add` (staging files)
  - `git commit`, `git commit --amend` (creating or amending commits)
  - `git rm`, `git mv`, `git reset` (removing/moving/resetting files or changes)
  - `git rebase -i` or any interactive rebase
  - `git push` (pushing commits to a remote)
- You may run non-destructive, read-only git commands for inspection and context, such as `git status`, `git diff`, `git log`, and `git blame`.
- If a change requires committing, ask the user for explicit permission and include a concise commit message proposal before performing any commit-related actions.
