# mlkit

A Python machine learning toolkit. Requires Python ≥ 3.12.

**Docs:** [pymlkit.zerodtree.com](https://pymlkit.zerodtree.com/)


## Installation

### uv (recommended)

```bash
uv add git+https://github.com/zeroDtree/my_pkg_py
```

**With optional dependencies:**

```bash
uv add "git+https://github.com/zeroDtree/my_pkg_py[hard]"
```

- `hard`: deepspeed, flash-attn, bitsandbytes (CUDA / quantization)

<details>
<summary>Other install options</summary>

**Editable (local clone):**

```bash
git clone git@github.com:zeroDtree/my_pkg_py.git
uv add --editable my_pkg_py
```

**pip:**

```bash
pip install "mlkit @ git+https://github.com/zeroDtree/my_pkg_py.git"
# with optional deps:
pip install "mlkit[hard] @ git+https://github.com/zeroDtree/my_pkg_py.git"
```

</details>

## Examples

```bash
git clone https://github.com/zeroDtree/my_pkg_py.git
cd my_pkg_py
```

See the [`example/`](example/) directory for usage by task.

Each example is driven by Hydra. Checkpoints, logs, the MLflow database, and Hydra output are relative paths under `artifacts/` in the directory you start from. A run started in the example directory does not resume a run started at the repo root.

From the example directory:

```bash
cd example/butterfly_edm
uv run python main.py                       # default config
uv run python main.py train.lr=1e-3         # override a single value
uv run python main.py --multirun gm.n_discretization_steps=100,200  # sweep
```

From the repo root, the same script writes to `<repo>/artifacts/` instead:

```bash
uv run python example/butterfly_edm/main.py
bash shell_script/mc-run-python.sh example/butterfly_edm/main.py
```

Config defaults live in the `config.yaml` (or `config_<variant>.yaml`) next to each script.


## Development

```bash
uv sync
# optional extras for CUDA tooling:
uv sync --extra hard
uv run nbstripout --install --attributes .gitattributes
```

`uv sync` installs package dependencies plus the `dev` group (lint, docs, notebooks).
`nbstripout` strips notebook cell outputs on commit. Run `--install` once per clone.
Verify with `uv run nbstripout --status`.

```bash
uv run ruff check
uv run ty check
uv run pytest
```

Run launchers from the repository root with `bash shell_script/...`. See [shell_script/README.md](shell_script/README.md).

## Docs

```bash
uv run mkdocs serve
# or on a custom address:
uv run mkdocs serve --dev-addr 127.0.0.1:8080
```
