.. ls-mlkit documentation master file, created by
   sphinx-quickstart on Mon Oct 27 13:51:32 2025.

ls-mlkit Documentation
======================

Welcome to **ls-mlkit**, a comprehensive machine learning toolkit that provides various utilities for deep learning, diffusion models, optimization, and data processing.


Installation
----------------

.. toctree::
   :maxdepth: 1
   :caption: Installation:

   installation


API
----------------

.. toctree::
   :maxdepth: 4
   :caption: API Reference:

   api/index

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


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

