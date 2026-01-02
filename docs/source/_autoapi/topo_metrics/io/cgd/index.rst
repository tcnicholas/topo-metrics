topo_metrics.io.cgd
===================

.. py:module:: topo_metrics.io.cgd


Functions
---------

.. autoapisummary::

   topo_metrics.io.cgd.parse_cgd
   topo_metrics.io.cgd.process_neighbour_list


Module Contents
---------------

.. py:function:: parse_cgd(filename: pathlib.Path | str) -> tuple[pymatgen.core.lattice.Lattice, list[str], numpy.typing.NDArray[numpy.floating] | None, list[list[Any]]]

   Parses a CGD file and extracts lattice, atoms, and edges.

   :param filename: The path to the CGD file.

   :returns: * **- lattice** (*The lattice of the network.*)
             * **- atom_labels** (*The labels of the atoms.*)
             * **- all_coords** (*The fractional coordinates of the atoms.*)
             * **- edges** (*The list of edges in the network.*)


.. py:function:: process_neighbour_list(edges: list[list[Any]], atom_frac_coords: numpy.typing.NDArray[numpy.floating] | None, atom_labels: list[str]) -> numpy.typing.NDArray[numpy.int_]

   Processes edges and determines node connectivity.

   :param edges: The list of edges in the network.
   :param atom_frac_coords: The fractional coordinates of the atoms.
   :param atom_labels: The labels of the atoms.

   :returns: * *The neighbour list of the network, where each row corresponds to a pair of*
             * *connected nodes and the applied image shift to the second node.*


