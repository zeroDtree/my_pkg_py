# mlkit

A Python machine learning toolkit. Requires Python ≥ 3.12.

**Docs:** [pymlkit.zerodtree.com](https://pymlkit.zerodtree.com/)

## Quick Start

```bash
uv add git+https://github.com/zeroDtree/my_pkg_py
```

See the [`example/`](example/) directory for usage by task.

Each example is driven by Hydra. Run from within the example directory:

```bash
cd example/butterfly_edm
uv run python main.py                       # default config
uv run python main.py train.lr=1e-3         # override a single value
uv run python main.py --multirun gm.n_discretization_steps=100,200  # sweep
```

Config defaults live in the `config.yaml` (or `config_<variant>.yaml`) next to each script.

## Installation

### uv (recommended)

```bash
uv add git+https://github.com/zeroDtree/my_pkg_py
```

**With optional dependencies:**

```bash
uv add "git+https://github.com/zeroDtree/my_pkg_py[bio]"
```

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
pip install "mlkit[bio] @ git+https://github.com/zeroDtree/my_pkg_py.git"
```

</details>

## Development

```bash
uv sync --group dev
uv run nbstripout --install --attributes .gitattributes
```

`nbstripout` strips notebook cell outputs on commit. Run `--install` once per clone.
Verify with `uv run nbstripout --status`.

## Docs

```bash
uv run mkdocs serve
# or on a custom address:
uv run mkdocs serve --dev-addr 127.0.0.1:8080
```
