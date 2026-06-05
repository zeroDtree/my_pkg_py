## 1. Install

### 1.1. By [uv](https://github.com/astral-sh/uv)

```bash
uv add git+https://git@github.com/zeroDtree/my_pkg_py
```

or
```bash
uv add git+ssh://git@github.com/zeroDtree/my_pkg_py
```

or
```bash
git clone git@github.com:zeroDtree/my_pkg_py.git
uv add --editable my_pkg_py
```


#### 1.1.1. With optional dependencies

```bash
uv add git+https://git@github.com/zeroDtree/my_pkg_py[bio]
```
or
```bash
uv add git+https://git@github.com/zeroDtree/my_pkg_py --optional bio
```

### 1.2. By pip (A virtual environment is recommended)
```bash
pip install "ls_mlkit @ git+https://github.com/zeroDtree/my_pkg_py.git"
```

#### 1.2.1. With optional dependencies

```bash
pip install "ls_mlkit[bio] @ git+https://github.com/zeroDtree/my_pkg_py.git"
```

## 2. Development

After cloning the repository, install dev dependencies and register the Git filter for notebooks (strips cell outputs on commit; `.gitattributes` is already in the repo):

```bash
uv sync --group dev
uv run nbstripout --install --attributes .gitattributes
```

Verify with `uv run nbstripout --status`. Run `--install` once per clone (or after recreating `.git/config`).

## 3. Docs

### 3.1. Generate documentation locally (uv is required)

```bash
uv run mkdocs serve
```

To use a custom address:

```bash
uv run mkdocs serve --dev-addr 127.0.0.1:8080
```

## 4. Example

`example` directory contains examples of how to use the package for some specific tasks.
