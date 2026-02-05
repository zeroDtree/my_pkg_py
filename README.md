- [1. Install](#1-install)
	- [1.1. By uv](#11-by-uv)
		- [1.1.1. With optional dependencies](#111-with-optional-dependencies)
	- [1.2. by pip (A virtual environment is recommended)](#12-by-pip-a-virtual-environment-is-recommended)
		- [1.2.1. With optional dependencies](#121-with-optional-dependencies)
- [2. Docs](#2-docs)
	- [2.1. Generate documentations locally (uv is required)](#21-generate-documentations-locally-uv-is-required)
- [3. Example](#3-example)


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

### 1.2. by pip (A virtual environment is recommended)
```bash
pip install "ls_mlkit @ git+https://github.com/zeroDtree/my_pkg_py.git"
```

#### 1.2.1. With optional dependencies

```bash
pip install "ls_mlkit[bio] @ git+https://github.com/zeroDtree/my_pkg_py.git"
```

## 2. Docs

Online [documents](https://zeroDtree.github.io/my_pkg_py/) are generated using Sphinx.

### 2.1. Generate documentations locally (uv is required)

```bash
make -C docs html
```

## 3. Example

`example` directory contains examples of how to use the package for some specific tasks.
