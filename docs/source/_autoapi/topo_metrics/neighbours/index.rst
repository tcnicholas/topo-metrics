topo_metrics.neighbours
=======================

.. py:module:: topo_metrics.neighbours


Functions
---------

.. autoapisummary::

   topo_metrics.neighbours.graph_edges_by_cutoff
   topo_metrics.neighbours.autoreduce_neighborlist


Module Contents
---------------

.. py:function:: graph_edges_by_cutoff(atoms: ase.Atoms, cutoff: float = 0.0, pair_cutoffs: dict[tuple[str, str], float] | None = None, one_based: bool = True)

   Build periodic edges with a global cutoff or element-specific cutoffs.

   :param atoms: Input structure (ase.Atoms).
   :param cutoff: Global distance cutoff used when `pair_cutoffs` is None.
                  If `pair_cutoffs` is provided, this acts as the default cutoff
                  for element pairs not present in `pair_cutoffs`.
                  Set to 0.0 to effectively forbid unspecified pairs.
   :param pair_cutoffs: Per-element-pair cutoffs, e.g. {("Si", "O"): 2.1, ("Al", "O"): 2.0}.
                        Pairs are treated as unordered; ("O", "Si") is the same as ("Si", "O").
                        If None, a single global cutoff is used for all pairs.
   :param one_based: If True, returned atom indices are 1-based. If False, they are 0-based.

   :returns: ndarray of shape (n_edges, 5). Columns: [i, j, sx, sy, sz],
             where (sx, sy, sz) are integer cell shifts.
   :rtype: edges


.. py:function:: autoreduce_neighborlist(cart_coords: numpy.typing.NDArray[numpy.float64], frac_coords: numpy.typing.NDArray[numpy.float64] | list[None], symbols: list[str], edges: numpy.typing.NDArray[numpy.int_], remove_types: Iterable[Any] | None = None, remove_degree2: bool = False) -> tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64], list[str], numpy.typing.NDArray[numpy.int_], numpy.typing.NDArray[numpy.int_]]

   Simplify a periodic bonded graph by contracting out selected atoms.

   :param frac_coords: Fractional coordinates of all atoms.
   :param symbols: Atomic symbols (length N).
   :param edges: Columns: i+1, j+1, Sx, Sy, Sz (1-based atom indices).
   :param remove_types: If not None, atoms whose symbol is in this set are removed and their
                        neighbors are connected together (clique over neighbors).
                        Example: {"O"}.
   :param remove_degree2: If True, atoms that are 2-connected are also removed (in addition to
                          any atoms in `remove_types`).

   :returns: * **new_frac_coords** (*(N_keep,3) ndarray*)
             * **new_symbols** (*list[str] length N_keep*)
             * **new_edges** (*(M_new,5) int ndarray*) -- Same format as input `edges` (1-based indices).
             * **old_to_new** (*(N,) ndarray of int*) -- Mapping from old atom index (0-based) to new index (0-based). -1 for
               removed atoms.


