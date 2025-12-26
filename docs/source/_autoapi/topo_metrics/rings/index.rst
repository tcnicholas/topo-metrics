topo_metrics.rings
==================

.. py:module:: topo_metrics.rings


Classes
-------

.. autoapisummary::

   topo_metrics.rings.RingSizeCounts
   topo_metrics.rings.Ring


Functions
---------

.. autoapisummary::

   topo_metrics.rings.get_ordered_node_list
   topo_metrics.rings.node_list_to_ring


Module Contents
---------------

.. py:class:: RingSizeCounts

   .. py:attribute:: sizes
      :type:  numpy.typing.NDArray[numpy.int_]


   .. py:attribute:: counts
      :type:  numpy.typing.NDArray[numpy.int_]


.. py:class:: Ring

   Bases: :py:obj:`NamedTuple`


   .. py:attribute:: nodes
      :type:  list[topo_metrics.topology.Node]

      The node IDs that form the ring. Neighbouring nodes are connected by an
      edge, and the last node is connected to the first node.


   .. py:attribute:: angle
      :type:  tuple[tuple[int, tuple[int]]]

      Neighbouring nodes about the central node, listed with node ID and image.


   .. py:property:: size
      :type: int


      Ring size.


   .. py:method:: geometry() -> topo_metrics.ring_geometry.RingGeometry

      Get the ring geometry object.



.. py:function:: get_ordered_node_list(nodes_and_images: list[tuple[int, numpy.typing.NDArray[numpy.int_]]], central_node_id: int) -> list[tuple[int, numpy.typing.NDArray[numpy.int_]]]

   Sorts a cyclic list of nodes and images, placing the central_node_id first
   and preserving cyclic order while ensuring a deterministic ordering.

   :param nodes_and_images: A list of node IDs and image shifts.
   :param central_node_id: The node ID of the central node.

   :rtype: A sorted list of node IDs and image shifts.


.. py:function:: node_list_to_ring(topology: topo_metrics.topology.Topology, node_list: list[tuple[int, numpy.typing.NDArray[numpy.int_]]], central_node_id: int) -> Ring

   Convert a list of node IDs and image shifts to a Ring object.

   :param topology: The topology object containing the nodes.
   :param node_list: A list of node IDs and image shifts.
   :param central_node_id: The node ID of the central node.

   :rtype: A Ring object with the nodes in the correct positions.


