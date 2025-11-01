.. ls-mlkit documentation master file, created by
   sphinx-quickstart on Mon Oct 27 13:51:32 2025.

ls-mlkit Documentation
======================

Welcome to **ls-mlkit**, a comprehensive machine learning toolkit that provides various utilities for deep learning, diffusion models, optimization, and data processing.

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/zeroDtree/my_pkg_py.git
   cd my_pkg_py
   pip install -e .


Features Overview
-----------------

   **Pipeline**
   ``BasePipeline``, ``DistributedPipeline``, ``MyDistributedPipeline``

   **Diffuser**
   ``DDPM``, ``DDIM``, and ``SO(3)VPSDE``

   **Scheduler**
   ``cosine(_with_warmup)``, ``linear(_with_warmup)``, ``constant``, ``cosine_annealing``

   **Optimizers**
   ``KFA``, ``SAM``

   **Flow Matching**
   ``EuclideanFlow``

   **Datasets**
      Numerical datasets:
         ``IrisDataset``
      Image datasets:
         ``MNIST``, ``FashionMNIST``, ``CIFAR10``, ``CIFAR100``
      Language datasets:
         ``LDADataset``, ``MT19937``, ``RegularLanguageDataset``, ``meta-math/MetaMathQA``, ``gsm8k``, ``glue/sst2``, ``m-a-p/CodeFeedback-Filtered-Instruction``, ``silk-road/Wizard-LM-Chinese-instruct-evol``, ``tatsu-lab/alpaca``

   **Models**
   ``LongLinearModel``, ``CausalLanguageModelForAuto``

   **Utils**
   Various tools

Templates
-------------

`template` directory contains templates for some specific tasks, which maybe useful for you to understand (familiarize yourself with) the usage of ls-mlkit.


.. toctree::
   :maxdepth: 1
   :caption: Installation:

   installation

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

