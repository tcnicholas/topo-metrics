topo_metrics.topology
=====================

.. py:module:: topo_metrics.topology


Classes
-------

.. autoapisummary::

   topo_metrics.topology.RingsResults
   topo_metrics.topology.Topology
   topo_metrics.topology.Node


Functions
---------

.. autoapisummary::

   topo_metrics.topology.get_all_node_frac_coords


Module Contents
---------------

.. py:class:: RingsResults

   Bases: :py:obj:`NamedTuple`


   .. py:attribute:: depth
      :type:  int

      The depth to which the rings were searcher for.


   .. py:attribute:: rings_are_strong
      :type:  bool

      Whether the rings were filtered to strong rings only.


   .. py:attribute:: ring_size_count
      :type:  topo_metrics.rings.RingSizeCounts

      The number of rings of a given size, of shape ``(**, 2)`` where the first
      column indicates the ring size, and the second indicates the number of rings
      of that size.


   .. py:attribute:: clusters
      :type:  list[topo_metrics.clusters.Cluster]

      Each node can be characterised in terms of the rings in which it
      participates. This can be summarised using common metrics such as Vertex
      Symbols.


.. py:class:: Topology

   A class detailing the topology of a network, based on nodes and edges.


   .. py:attribute:: nodes
      :type:  list[Node]


   .. py:attribute:: edges
      :type:  numpy.typing.NDArray[numpy.int_]


   .. py:attribute:: lattice
      :type:  pymatgen.core.lattice.Lattice | None
      :value: None



   .. py:attribute:: properties
      :type:  dict[Hashable, Any]


   .. py:method:: from_ase(ase_atoms: ase.Atoms, cutoff: float = 0.0, pair_cutoffs: dict[tuple[str, str], float] | None = None, remove_types: Iterable[Any] | None = None, remove_degree2: bool = False) -> Topology
      :classmethod:


      Creates a Topology object from an ASE Atoms object.

      :param ase_atoms: The ASE Atoms object representing the structure.

      :rtype: A Topology object representing the network as nodes and edges.



   .. py:method:: from_cgd(filename: pathlib.Path | str) -> Topology
      :classmethod:


      Parses and loads a CGD file with an adjacency matrix.

      :param filename: The path to the CGD file.

      :rtype: A Topology object representing the network as nodes and edges.



   .. py:method:: from_conflink(filename: str, node_type: str = 'Si', index: int | None = None) -> Topology
      :classmethod:


      Parses and loads a conflink file.

      :param filename: The path to the conflink file.
      :param node_type: The type to assign to all nodes in the topology.

      :rtype: A Topology object representing the network as nodes and edges.



   .. py:method:: get_rings(depth: int = 12) -> list[topo_metrics.ring_geometry.RingGeometry]

      Computes or retrieves unique rings in the network.

      :param depth: The maximum depth to search for rings.

      .. rubric:: Notes

      - In the previous implementation, this method returned the clusters of
        rings at each node. This is obtained instead via the `get_clusters`
        method. This method now returns all unique rings in the network as
        RingGeometry objects.



   .. py:method:: get_clusters(depth: int = 12, strong: bool = False) -> RingsResults

      Computes or retrieves ring statistics for the network.

      :param depth: The maximum depth to search for rings.
      :param strong: Whether to filter the rings to strong rings only.

      :rtype: A dictionary containing the ring statistics.



   .. py:method:: get_topological_genome() -> str

      Returns a the topology code for the framework.

      .. rubric:: Notes

      - The topological genome is a finite series of numbers that is provably
        unique for each net.
      - It can be comptued in polynomial time with respect to the size of the
        net.



   .. py:method:: get_coordination_sequences(max_shell: int = 10, node_ids: Iterable[int] | int | None = None) -> numpy.typing.NDArray[numpy.int_]

      Return coordination sequences for specified nodes.

      :param max_shell: The maximum shell to compute coordination sequences to.
      :param node_ids: The node IDs for which to compute coordination sequences. If None,
                       coordination sequences for all nodes are returned.



   .. py:property:: cartesian_coordinates
      :type: numpy.typing.NDArray[numpy.floating]


      Return the Cartesian positions of all nodes in the network.


   .. py:property:: fractional_coordinates
      :type: numpy.typing.NDArray[numpy.floating]


      Return the fractional positions of all nodes in the network.


.. py:class:: Node

   A representation of a node in a network.


   .. py:attribute:: node_id
      :type:  int


   .. py:attribute:: node_type
      :type:  str | None
      :value: 'Si'



   .. py:attribute:: frac_coord
      :type:  numpy.typing.NDArray[numpy.floating] | None
      :value: None



   .. py:attribute:: cart_coord
      :type:  numpy.typing.NDArray[numpy.floating] | None
      :value: None



   .. py:attribute:: is_shifted
      :type:  bool
      :value: False



   .. py:method:: apply_image_shift(lattice: pymatgen.core.lattice.Lattice, image_shift: numpy.typing.NDArray[numpy.int_]) -> Node

      Apply the image shift to this node and return a new Node object.

      :param lattice: The lattice object for the network.
      :param image_shift: The shift vector to apply to the node coordinates.

      :rtype: A new Node object with the shifted coordinates.



.. py:function:: get_all_node_frac_coords(nodes: list[Node]) -> numpy.typing.NDArray[numpy.floating]

   Return the fractional coordinates of all nodes in the network.


