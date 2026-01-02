topo_metrics.io.conflink
========================

.. py:module:: topo_metrics.io.conflink


Classes
-------

.. autoapisummary::

   topo_metrics.io.conflink.Configuration


Functions
---------

.. autoapisummary::

   topo_metrics.io.conflink.parse_conflink


Module Contents
---------------

.. py:class:: Configuration

   A configuration parsed from a conflink file.


   .. py:attribute:: positions
      :type:  numpy.typing.NDArray[numpy.float64]


   .. py:attribute:: box_size
      :type:  float


   .. py:attribute:: connections
      :type:  dict


   .. py:method:: to_nodes_edges_lattice(node_type: str = 'Si') -> dict

      Convert this configuration to a Topology object.



.. py:function:: parse_conflink(filename: str) -> list[Configuration]

   Parse one or more configurations from a conflink file.


