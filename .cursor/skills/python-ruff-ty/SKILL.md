---
name: python-ruff-ty
description: Run Ruff lint/format fixes and ty type checks via uv, bootstrap ruff.toml and ty.toml from skill defaults when missing, and re-run until clean. Use when finalizing Python changes, fixing lint or type errors, or when the user mentions ruff, ty, ruff check, ty check, or Python quality gates.
disable-model-invocation: true
---

# Python Ruff + ty Quality Gate

## When to apply

- User asks to lint, fix lint, type-check, or pass ruff/ty.
- Before marking Python work complete (even if workspace rules already mention checks).

## Command rules

Always run tools through **uv** (never bare `ruff` / `ty` / `python`):

```bash
uv run ruff check
uv run ruff check --fix
uv run ruff format
uv run ty check
```

If `uv` is unavailable, stop and ask the user to install uv or use an activated `.venv` with `ruff` and `ty` installed.

## Workflow

Copy and track:

```
Progress:
- [ ] Locate project root (nearest pyproject.toml or repo root)
- [ ] Ensure ruff + ty config and dev dependencies exist
- [ ] Auto-fix with Ruff (check --fix, then format)
- [ ] Run ty check; fix remaining issues manually
- [ ] Re-run all checks until both pass
```

### Step 1: Locate project root

Walk up from edited files until you find `pyproject.toml`. If none exists, use the workspace root where Python code lives.

### Step 2: Config and dependencies

**Default config templates** (copy from the skill directory, do not embed in `pyproject.toml`):

| Template | Source |
|----------|--------|
| Ruff | `.cursor/skills/python-ruff-ty/ruff.toml` |
| ty | `.cursor/skills/python-ruff-ty/ty.toml` |

**Detect config** (any of these counts):

| Tool | Config locations |
|------|------------------|
| Ruff | `ruff.toml` at project root, or `[tool.ruff]` in `pyproject.toml` |
| ty | `ty.toml` at project root, or `[tool.ty]` in `pyproject.toml` |

**If Ruff config is missing** (no `ruff.toml` and no `[tool.ruff]` in `pyproject.toml`), copy the template to the project root:

```bash
cp .cursor/skills/python-ruff-ty/ruff.toml ruff.toml
```

**If ty config is missing** (no `ty.toml` and no `[tool.ty]` in `pyproject.toml`), copy the template to the project root:

```bash
cp .cursor/skills/python-ruff-ty/ty.toml ty.toml
```

Resolve paths relative to the workspace root. After copying, adjust only when the repo layout requires it (e.g. add `src = ["src"]` in `ruff.toml` or ty `root` when code lives under `src/`).

**If `pyproject.toml` is missing**, create a minimal one for project metadata and dev dependencies only—**do not** add `[tool.ruff]` or `[tool.ty]`:

```toml
[project]
name = "project"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = []

[dependency-groups]
dev = ["ruff>=0.9", "ty>=0.0.1"]
```

**If `pyproject.toml` exists**, leave its tool sections untouched; only copy standalone `ruff.toml` / `ty.toml` when that tool has no config yet.

**Dependencies**: Ensure dev tools are installable:

```bash
uv add --dev ruff ty
```

Skip if `pyproject.toml` already lists them under `[dependency-groups] dev`, `[project.optional-dependencies]`, or `[tool.uv] dev-dependencies`.

### Step 3: Ruff auto-fix

Run in order from project root:

```bash
uv run ruff check --fix .
uv run ruff format .
```

Scope `.` to changed paths when the repo is huge and the user only touched a few files.

If `ruff check` still reports unfixable violations, fix them in code (do not use `# noqa` unless the user asks or the project already uses suppressions consistently).

### Step 4: ty type check

```bash
uv run ty check
```

ty does not auto-fix. Address diagnostics by correcting types, imports, or signatures. Use [ty suppression comments](https://docs.astral.sh/ty/suppression/) only when justified and aligned with project style.

### Step 5: Verify (blockers)

Both must exit 0 before the task is done:

```bash
uv run ruff check
uv run ty check
```

Re-run Step 3–5 after each fix batch until clean.

## Decision: create vs extend config

| Situation | Action |
|-----------|--------|
| No Ruff config | Copy `.cursor/skills/python-ruff-ty/ruff.toml` → project `ruff.toml` |
| No ty config | Copy `.cursor/skills/python-ruff-ty/ty.toml` → project `ty.toml` |
| No `pyproject.toml` | Create minimal metadata-only file + `uv add --dev ruff ty` |
| `ruff.toml` / `ty.toml` or `[tool.ruff]` / `[tool.ty]` already present | Do not duplicate; respect existing config format |
| User has strict existing config | Never weaken rule sets; only add missing tooling |

## Output to the user

When finished, briefly report:

- Whether `ruff.toml` / `ty.toml` were copied or existing config was reused
- Commands run
- Any remaining manual fixes (if blocked)

When checks fail, quote the specific rule/diagnostic and file path—do not paraphrase error codes incorrectly.
