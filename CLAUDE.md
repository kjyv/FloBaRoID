## Information sources

- IMPORTANT: For finding code references, call sites, and definitions, ALWAYS use `sg` (ast-grep)
  via Bash, NEVER use the Grep tool or run python import scripts. Example: `sg run -p 'function_name'
  -l python` for structural code search, `sg run -p '$X.method($ARG)' -l python` for method calls
- IMPORTANT: For refactoring or removing code across multiple files, ALWAYS use
  `ast-grep run --pattern 'foo' --rewrite 'bar' --lang python -U` for bulk renames instead of
  making individual Edit calls. Reserve Edit for surgical, logic-changing modifications.

## Development rules

- Absolutely don't remove code comments when there is no reason to do so (functionality is the same
  and the code describes what is happening)
- Don't remove "TODO:" comments unless you implemeted or fixed the thing mentioned
- Always use type annotations for better code clarity; don't use Union or Optional anymore
- Don't attempt to fix mypy errors using `# type: ignore`, fix them properly
- Don't put import or exports in __init__.py files without asking and getting confirmation
- Always put imports at the top of the file, never within methods
- When you need e.g. a company name for a plant, don't split the plant_slug, always get the
  company properly from DB fields. This is just an example and is to be followed for all slugs
  that contain other data

- The project uses pyproject.toml and uv for packages and virtual env management
- After finishing a task, you must always run
    - `uv run ruff check . --fix && uv run ruff format . && uv run mypy 2>&1`
    - this means: do not run mypy on single files (always use `uv run mypy`)
    - this means: do not run ruff check without `--fix`
    - do NOT run ast syntax checks (e.g. python -c "import ast; ...") — ruff handles syntax checking,
      so this would be redundant

- You're not allowed to use git stash or git stash pop on your own, ask about it if you need it.

## Testing preferences

- Write all Python tests as `pytest` style functions, not unittest classes
- Use descriptive function names starting with `test_`
- Prefer fixtures over setup/teardown methods
- Use assert statements directly, not self.assertEqual
- Always add a docstring to new tests (and also all other methods you create)

## Output

- If while analyzing data and from discussions with the user you find something interesting and
  noteworthy, write it down in an analysis document
