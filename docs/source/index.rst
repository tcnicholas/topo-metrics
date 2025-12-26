.. toctree::
   :hidden:
   :maxdepth: 1

   Examples <examples/index>
   Basics <basics/root>
   API Reference <_autoapi/index>

TopoMetrics
===========

**Date:** |today| — **Author:** `Thomas C Nicholas <https://tcnicholas.github.io/>`__ — **Version:** |release|

``topo-metrics`` is a Python package for calculating topological metrics of periodic materials.

The core functionality is taking periodic atomic structures and turning them 
into graphs, and then computing a range of graph and topology-based metrics 
(rings, clusters, coordination sequences) that are useful for analysing porous 
materials, frameworks, and other periodic networks.


Installation
------------

Install ``topo-metrics`` using ``pip`` (preferably in a fresh environment):

.. code-block:: bash

   conda create -n topo-metrics python=3.11 -y
   conda activate topo-metrics
   pip install topo-metrics

Quick start
-----------

Here's a tiny “this does something” example you can later replace with real code:

.. code-block:: python

   from ase.io import read
   import topo_metrics as tm

   # read in a structure with ASE
   atoms = read("mystructure.cif")

   # build the graph and compute ring statistics
   graph = tm.Topology.from_ase(ase_atoms=atoms, cutoff=1.7)
   ring_stats = tm.rings.compute_ring_statistics(graph)

   print(ring_stats.summary())

Where to go next
----------------

- :doc:`Basics <basics/root>` - basic definitions and concepts used throughout TopoMetrics.
- :doc:`Examples <examples/index>` – end-to-end, copy-and-pasteable examples.
- :doc:`API Reference <_autoapi/index>` – full auto-generated API docs.

Project links
-------------

- `GitHub repository <https://github.com/tcnicholas/topo-metrics>`__
- `PyPI project page <https://pypi.org/project/topo-metrics/>`__
