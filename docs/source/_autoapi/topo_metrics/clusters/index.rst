topo_metrics.clusters
=====================

.. py:module:: topo_metrics.clusters


Classes
-------

.. autoapisummary::

   topo_metrics.clusters.Cluster


Functions
---------

.. autoapisummary::

   topo_metrics.clusters.smallest_ring_size
   topo_metrics.clusters.largest_ring_size
   topo_metrics.clusters.number_of_rings
   topo_metrics.clusters.get_unique_angles
   topo_metrics.clusters.number_of_angles
   topo_metrics.clusters.get_ring_sizes
   topo_metrics.clusters.shares_edge
   topo_metrics.clusters.get_opposite_angles
   topo_metrics.clusters.only_smallest_ring_size
   topo_metrics.clusters.get_ring_size_by_angle
   topo_metrics.clusters.ring_list_to_cluster
   topo_metrics.clusters.get_clusters
   topo_metrics.clusters.get_vertex_symbol
   topo_metrics.clusters.get_carvs_std_dev
   topo_metrics.clusters.get_carvs_vector


Module Contents
---------------

.. py:class:: Cluster

   Bases: :py:obj:`NamedTuple`


   .. py:attribute:: central_node_id
      :type:  int

      The node ID of the central node in the cluster.


   .. py:attribute:: rings
      :type:  list[topo_metrics.rings.Ring]

      The rings that form the cluster.


.. py:function:: smallest_ring_size(cluster: Cluster) -> int

   Get the smallest ring size in the ``cluster``.


.. py:function:: largest_ring_size(cluster: Cluster) -> int

   Get the largest ring size in the ``cluster``.


.. py:function:: number_of_rings(cluster: Cluster) -> int

   Get the number of rings in the ``cluster``.


.. py:function:: get_unique_angles(cluster: Cluster) -> int

   Get the number of angles in the ``cluster``.


.. py:function:: number_of_angles(cluster: Cluster) -> int

   Get the number of angles in the ``cluster``.


.. py:function:: get_ring_sizes(cluster: Cluster) -> list

   Get the ring sizes in the ``cluster``.


.. py:function:: shares_edge(angle1: tuple, angle2: tuple) -> bool

   Checks if two angles share an edge.

   :param angle1: The first angle.
   :param angle2: The second angle.

   :rtype: True if the angles share an edge, False otherwise.


.. py:function:: get_opposite_angles(angles: list) -> list

   Finds the opposite angles in a cluster.

   :param angles: A list of angles in the cluster.

   :rtype: A list of opposite angle pairs.


.. py:function:: only_smallest_ring_size(ring_sizes: list[int]) -> list[int]

   Retains only the smallest ring size in this list. If there are multiple
   smallest ring sizes, they are all retained.

   :param ring_sizes: A list of ring sizes.

   :rtype: A list of the smallest ring sizes.


.. py:function:: get_ring_size_by_angle(cluster: Cluster, all_rings: bool = False) -> dict[int, list[int]]

   Get the ring sizes by angle in a cluster.

   :param cluster: The cluster to get the ring sizes by angle of.
   :param all_rings: Whether to consider all rings in the cluster or just the smallest ring
                     at each angle.

   :rtype: A dictionary of ring sizes by angle.


.. py:function:: ring_list_to_cluster(topology: topo_metrics.topology.Topology, ring_list: list[list[tuple[int, numpy.typing.NDArray[numpy.int_]]]], central_node_id: int) -> Cluster

   Convert a list of rings to a Cluster object.

   :param topology: The topology object containing the nodes.
   :param ring_list: A list of lists of node IDs and image shifts.
   :param central_node_id: The node ID of the central node.

   :rtype: A Cluster object with the rings in the correct positions.


.. py:function:: get_clusters(topology: topo_metrics.topology.Topology, all_ring_list: list[list[list[tuple[int, numpy.typing.NDArray[numpy.int_]]]]])

   Convert a list of lists of rings to a list of Cluster objects.

   :param topology: The topology object containing the nodes.
   :param all_ring_list: A list of lists of lists of node IDs and image shifts.

   :rtype: A list of Cluster objects with the rings in the correct positions.


.. py:function:: get_vertex_symbol(clusters: Cluster | list[Cluster]) -> topo_metrics.symbols.VertexSymbol | list[topo_metrics.symbols.VertexSymbol]

   Obtain the vertex symbol(s) for one or more clusters, storing both the
   `all_rings=False` and `all_rings=True` vectors. If a list of clusters is
   provided, this function returns a list of vertex symbols in the same order.

   :param clusters: Either a single cluster, or a list of cluster objects from which to
                    obtain vertex symbols.

   :returns: * *VertexSymbol or list of VertexSymbol*
             * *- If a single Cluster is passed, returns a single VertexSymbol.*
             * *- If a list of Cluster objects is passed, returns a list of VertexSymbols.*


.. py:function:: get_carvs_std_dev(carvs_vectors: numpy.typing.NDArray[numpy.int_]) -> float

   Compute the standard deviation amongst the node environments.

   :param carvs_vectors: The CARVS vectors for each cluster in the network.

   :rtype: The standard deviation of the CARVS vectors.


.. py:function:: get_carvs_vector(cluster: list[Cluster] | Cluster, max_size: int | None = None, return_per_cluster: bool = False, unique: bool = False) -> topo_metrics.symbols.CARVS | numpy.typing.NDArray[numpy.floating]

   Get the Cummulative All-Rings Vertex Symbol (CARVS) vector.

   If the input is a ``Cluster`` object, the CARVS is calculated for that
   specific cluster. If the input is a list of ``Cluster`` objects, the CARVS
   vector is calculated as an average over all ``clusters`` in the entire
   network, by calling the ``get_carvs_vector`` method for each cluster.

   :param cluster: The input topology object (either a ``Cluster`` or a list of ``Cluster``
                   objects).
   :param max_size: The maximum ring size to consider. If not specified, the maximum ring
                    size in the network is used.
   :param return_per_cluster: Whether to return the CARVS vector for each cluster in the network, or
                              to return the average CARVS vector over all clusters.
   :param unique: If returning the CARVS vector for each cluster, whether to return only
                  unique CARVS vectors.

   :rtype: The CARVS vector.


