---
description: "Use when: running Python commands, scripts, or tests. Ensures uv or venv is always used as the Python environment manager."
applyTo: "**/*.py"
---

# Python Environment

Always use one of the following when running Python commands or scripts:

- Prefix commands with `uv run`: `uv run python script.py`, `uv run pytest`, etc.
- Or activate the virtual environment first: `source .venv/bin/activate`

```bash
# ✅ GOOD
uv run python src/ls_mlkit/util/scheduler.py
uv run pytest tests/

# ✅ ALSO GOOD
source .venv/bin/activate
python src/ls_mlkit/util/scheduler.py

# ❌ BAD
python script.py
pip install package
```

Prefer `uv run` for one-off commands and `source .venv/bin/activate` for interactive sessions.
