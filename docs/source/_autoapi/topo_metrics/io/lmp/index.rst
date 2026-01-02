topo_metrics.io.lmp
===================

.. py:module:: topo_metrics.io.lmp


Attributes
----------

.. autoapisummary::

   topo_metrics.io.lmp.RingLocalLink
   topo_metrics.io.lmp.GlobalLink


Classes
-------

.. autoapisummary::

   topo_metrics.io.lmp.PackedRings


Functions
---------

.. autoapisummary::

   topo_metrics.io.lmp.cycle_edges
   topo_metrics.io.lmp.set_types_by_ring_and_element
   topo_metrics.io.lmp.adjacency_to_ase_bonds_array
   topo_metrics.io.lmp.pack_rings_to_ase
   topo_metrics.io.lmp.add_ring_bonds_for_lammps_full
   topo_metrics.io.lmp.write_rings_lammps_full


Module Contents
---------------

.. py:data:: RingLocalLink

   



   Ring-local link:

       (ri, li, rj, lj, bond_type)

   where ri, rj are ring indices (0-based), li, lj are local indices within the
   rings (0-based), and bond_type is the LAMMPS bond type integer.

.. py:data:: GlobalLink

   



   Global link:

       (gi, gj, bond_type)

   where gi, gj are global atom indices (0-based), and bond_type is the LAMMPS bond
   type integer.

.. py:class:: PackedRings

   .. py:attribute:: atoms
      :type:  ase.Atoms


   .. py:attribute:: ring_ranges
      :type:  list[range]


   .. py:attribute:: ring_species
      :type:  list[str]


.. py:function:: cycle_edges(indices_0based: Sequence[int]) -> list[tuple[int, int]]

.. py:function:: set_types_by_ring_and_element(packed: PackedRings) -> dict[tuple[int, str], int]

   Assign LAMMPS atom types so that each (ring_index, element_symbol) is a
   unique type.

   :returns: mapping[(ring_idx, symbol)] = type_id  (type_id is 1-based)


.. py:function:: adjacency_to_ase_bonds_array(neigh: list[list[tuple[int, int]]]) -> numpy.ndarray

.. py:function:: pack_rings_to_ase(rings: Sequence, *, cell: numpy.ndarray | None = None, pbc: bool | tuple[bool, bool, bool] = False, images: numpy.ndarray | None = None, image_convention: str = 'subtract') -> PackedRings

   Packs a sequence of RingGeometry into a single ASE Atoms object.


.. py:function:: add_ring_bonds_for_lammps_full(packed: PackedRings, *, ring_bond_type: int | Sequence[int] = 1, links_ring_local: Iterable[RingLocalLink] = (), links_global: Iterable[GlobalLink] = (), mol_ids: Sequence[int] | None = None, charges: Sequence[float] | None = None) -> ase.Atoms

   Mutates packed.atoms in-place and returns it.


.. py:function:: write_rings_lammps_full(rings: Sequence, filepath: str, *, cell: numpy.ndarray | None = None, pbc: bool | tuple[bool, bool, bool] = False, images: numpy.ndarray | None = None, image_convention: str = 'subtract', ring_bond_type: int | Sequence[int] = 1, links_ring_local: Iterable[RingLocalLink] = (), links_global: Iterable[GlobalLink] = (), masses: bool = True) -> PackedRings

   Returns PackedRings for debugging (atom ordering, ring index ranges).


