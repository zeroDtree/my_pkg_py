Installation Guide
==================

Requirements
------------

- Python >= 3.12
- PyTorch (recommended latest stable version)
- CUDA (optional, for GPU acceleration)

Dependencies
------------

The package automatically installs the following dependencies:

.. code-block:: python

   dependencies = [
      "datasets",
      "exrex",
      "numpy",
      "omegaconf",
      "pandas",
      "peft",
      "psutil",
      "setuptools",
      "torchdiffeq",
      "torchviz",
      "tqdm",
      "transformers",
      "wandb",
      "bitsandbytes",
      "safetensors",
      "torchvision",
      "biotite",
      "pillow",
      "diffusers",
      "einops",
      "tabulate",
      "overrides",
   ]

Installation Methods
--------------------

From Source (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/zeroDtree/my_pkg_py.git
   cd my_pkg_py
   pip install -e .

or

.. code-block:: bash

   pip install "ls_mlkit @ git+https://github.com/zeroDtree/my_pkg_py.git" --no-cache-dir

or

.. code-block:: bash

   pip install "ls_mlkit[bio] @ git+https://github.com/zeroDtree/my_pkg_py.git" --no-cache-dir