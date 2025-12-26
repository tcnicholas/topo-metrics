topo_metrics.paths
==================

.. py:module:: topo_metrics.paths


Attributes
----------

.. autoapisummary::

   topo_metrics.paths.jl
   topo_metrics.paths.RingStatistics


Functions
---------

.. autoapisummary::

   topo_metrics.paths.get_project_root
   topo_metrics.paths.get_project_src
   topo_metrics.paths.get_data_dir
   topo_metrics.paths.instantiate_julia


Module Contents
---------------

.. py:data:: jl

.. py:data:: RingStatistics
   :value: None


.. py:function:: get_project_root() -> pathlib.Path

   Returns the project root directory.


.. py:function:: get_project_src() -> pathlib.Path

   Returns the path to the src directory located at the project root.


.. py:function:: get_data_dir() -> pathlib.Path

   Returns the path to the data directory located at the project root.


.. py:function:: instantiate_julia() -> None

   Initialises Julia, activates the environment, and imports RingStatistics.


