Installation Guide
==================

Requirements
------------

- Python >= 3.11
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

Verification
------------

To verify your installation, run:

.. code-block:: python

   import ls_mlkit
   print("ls-mlkit installed successfully!")
