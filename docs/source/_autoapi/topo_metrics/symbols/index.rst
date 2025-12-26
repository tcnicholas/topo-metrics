topo_metrics.symbols
====================

.. py:module:: topo_metrics.symbols


Classes
-------

.. autoapisummary::

   topo_metrics.symbols.VertexSymbol
   topo_metrics.symbols.CARVS


Functions
---------

.. autoapisummary::

   topo_metrics.symbols.pad_carvs
   topo_metrics.symbols.pad_carvs_per_atom
   topo_metrics.symbols.get_all_topological_distances


Module Contents
---------------

.. py:class:: VertexSymbol

   Bases: :py:obj:`NamedTuple`


   Vertex Symbol (VS) representation.

   .. attribute:: vector

      The Vertex Symbol vector.

   .. attribute:: vector_all_rings

      The Vertex Symbol vector considering all rings.


   .. py:attribute:: vector
      :type:  list[list[int]]


   .. py:attribute:: vector_all_rings
      :type:  list[list[int]]


   .. py:method:: to_str(all_rings: bool = False) -> str

      Returns the string representation of the VertexSymbol.

      If `all_rings` is True, ring counts are grouped and formatted with
      multiplicity. Otherwise, only the smallest ring sizes are shown, with
      multiplicity for repeated values.

      :param all_rings: If True, ring counts are grouped and formatted with multiplicity.
                        Otherwise, only the smallest ring sizes are shown, with multiplicity
                        for repeated values.

      :rtype: A string representation of the VertexSymbol.



.. py:class:: CARVS

   Bases: :py:obj:`NamedTuple`


   Cummulative All-Rings Vertex Symbol (CARVS) vector.

   .. attribute:: vector

      The CARVS vector.

   .. attribute:: spread

      The standard deviation of the CARVS vectors in the network.

   .. attribute:: is_single_node

      True if the CARVS vector is for a single-node network, False otherwise.


   .. py:attribute:: vector
      :type:  numpy.typing.NDArray[numpy.floating]


   .. py:attribute:: spread
      :type:  float


   .. py:attribute:: is_single_node
      :type:  bool


   .. py:method:: from_list(carvs_list: Sequence[CARVS]) -> CARVS
      :classmethod:


      Construct a CARVS object from a list of CARVS objects,
      averaging the vectors and spreads.

      :param carvs_list: One or more CARVS objects to be averaged.

      :returns: * *A new CARVS object whose vector is the average of all input vectors,*
                * *whose spread is the average of all input spreads, and whose*
                * *'is_single_node' is True only if it is True for every entry in*
                * `carvs_list`.

      :raises ValueError: If `carvs_list` is empty or if the vectors in `carvs_list` do not
          all have the same length.



.. py:function:: pad_carvs(carvs_list: Sequence[CARVS]) -> Sequence[CARVS]

   Pad the vectors of a list of CARVS objects to the same length.

   :param carvs_list: A list of CARVS objects.

   :rtype: A list of CARVS objects with the vectors padded to the same length.


.. py:function:: pad_carvs_per_atom(all_carvs: list[numpy.typing.NDArray[numpy.int_]]) -> list[numpy.typing.NDArray[numpy.int_]]

   Pad the CARVs per atom to the same length.

   :param all_carvs: List of CARVs per atom.

   :rtype: List of padded CARVs per atom.


.. py:function:: get_all_topological_distances(carvs: list[CARVS]) -> numpy.ndarray

   Compute the topological distances between all pairs of CARVS objects.

   :param carvs: A list of CARVS objects.

   :returns: * *A square matrix of shape (n_points, n_points) containing the Euclidean*
             * *distances between all pairs of points.*


