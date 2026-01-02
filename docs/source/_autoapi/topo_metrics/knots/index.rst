topo_metrics.knots
==================

.. py:module:: topo_metrics.knots


Attributes
----------

.. autoapisummary::

   topo_metrics.knots.nb
   topo_metrics.knots.Array


Functions
---------

.. autoapisummary::

   topo_metrics.knots.writhe_method_1a
   topo_metrics.knots.writhe_method_1b
   topo_metrics.knots.writhe_method_2a
   topo_metrics.knots.writhe_method_2b
   topo_metrics.knots.linking_number_method_1a
   topo_metrics.knots.linking_number_method_1b
   topo_metrics.knots.lk_round
   topo_metrics.knots.linking_number_int
   topo_metrics.knots.is_linked_from_lk
   topo_metrics.knots.linking_number_pbc


Module Contents
---------------

.. py:data:: nb
   :value: None


.. py:data:: Array

.. py:function:: writhe_method_1a(points: numpy.typing.ArrayLike, *, closed: bool = True, eps: float = 1e-12) -> tuple[float, float]

   Method 1a: pairwise solid angles (Eqs 13, 15-16).


.. py:function:: writhe_method_1b(points: numpy.typing.ArrayLike, *, closed: bool = True, eps: float = 1e-12) -> tuple[float, float]

   Method 1b: analytic Gauss integral (Eqs 13, 24-25).


.. py:function:: writhe_method_2a(points: numpy.typing.ArrayLike, *, eps: float = 1e-12) -> float

   Method 2a: Wr = Twz + Wrz - Tw (Eqs 30-34), z-axis projection.
   Requires a closed chain.


.. py:function:: writhe_method_2b(points: numpy.typing.ArrayLike, *, eps: float = 1e-12) -> float

   Method 2b (le Bret-style):
       Wr = Wrz - Tw with a_i = k×s_i/|k×s_i| (Eqs 35-38), k = z-axis.
   Requires closed chain.


.. py:function:: linking_number_method_1a(ring1: numpy.typing.ArrayLike, ring2: numpy.typing.ArrayLike, *, eps: float = 1e-12, disjoint_tol: float | None = None, disjoint_rel: float = 0.001, return_nan_if_not_disjoint: bool = True) -> float

   Gauss linking number for two CLOSED polygonal rings (method 1a).

   Disjointness:
     - if disjoint_tol is None: uses auto disjoint_tol from disjoint_rel
     - if disjoint_tol <= 0: skips the disjointness check
     - if not disjoint: returns nan or raises return_nan_if_not_disjoint=False


.. py:function:: linking_number_method_1b(ring1: numpy.typing.ArrayLike, ring2: numpy.typing.ArrayLike, *, eps: float = 1e-12, disjoint_tol: float | None = None, disjoint_rel: float = 0.001, return_nan_if_not_disjoint: bool = True) -> float

   Gauss linking number for two CLOSED polygonal rings (method 1b analytic).

   Same disjointness behavior as method_1a.


.. py:function:: lk_round(lk: float, tol: float = 1e-06) -> tuple[int, bool]

   Return (rounded_int, ok) where ok means |lk-round(lk)| <= tol.


.. py:function:: linking_number_int(lk: float, tol: float = 1e-06) -> int

   Convert near-integer lk to integer.


.. py:function:: is_linked_from_lk(lk: float, *, tol: float = 1e-06) -> bool

   Determine if two rings are linked from linking number value.


.. py:function:: linking_number_pbc(ringA: Array, ringB: Array, *, cell: Array, pbc: tuple[bool, bool, bool] = (True, True, True), n_images: int = 1, method: str = '1a', eps: float = 1e-12, check_top_k: int | None = None, disjoint_tol: float | None = None, disjoint_rel: float = 0.001) -> tuple[float, tuple[int, int, int]]

   Compute Gauss linking number between ringA and ringB under PBC by scanning
   periodic images of ringB.

   Returns (best_lk, best_image_shift) where best_image_shift is integer n such
   that:

       ringB_shifted = ringB - (n @ cell)

   Candidate scoring / selection:
     - compute min segment-segment dist^2 for each shift
     - discard candidates with dist < disjoint_tol
     - among remaining, choose the shift that maximizes |round(lk)|,
       tie-break by smaller dist^2, then smaller residual |lk-round(lk)|


