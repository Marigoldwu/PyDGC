:github_url: https://github.com/Marigoldwu/pydgc

PyDGC Documentation
=================================

Introduction
------------

**PyDGC**, a flexible and extensible Python library  for deep graph clustering (DGC), is compatible with frameworks such as PyG and OGB. It supports the easy integration of new models and datasets, facilitating the rapid development, reproduction, and fair comparison of DGC methods.


What is DGC?
------------

Deep graph clustering, which aims to reveal the underlying graph structure and divide the nodes into different groups, has attracted intensive attention in recent years. 

More details can be found in the `survey <https://arxiv.org/abs/2211.12875>` paper.


Installation via PyPi
---------------------

.. code-block:: bash

   pip install pydgc

Installation for local development
----------------------------------

.. code-block:: bash

  git clone https://github.com/Marigoldwu/PyDGC.git
  cd PyDGC
  pip install -e .

.. toctree::
   :maxdepth: 2
   :caption: Package Reference

   api/pydgc

.. toctree::
   :maxdepth: 1
   :caption: External Resources

   external/resources
