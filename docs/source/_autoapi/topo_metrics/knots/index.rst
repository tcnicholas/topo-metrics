topo_metrics.knots
==================

.. py:module:: topo_metrics.knots


Attributes
----------

.. autoapisummary::

   topo_metrics.knots.Array


Functions
---------

.. autoapisummary::

   topo_metrics.knots.writhe_method_1a
   topo_metrics.knots.writhe_method_1b
   topo_metrics.knots.writhe_method_2a
   topo_metrics.knots.writhe_method_2b


Module Contents
---------------

.. py:data:: Array

.. py:function:: writhe_method_1a(points: numpy.typing.ArrayLike, *, closed: bool = True, eps: float = 1e-12) -> tuple[float, float]

   Method 1a: pairwise solid angles (Eqs 13, 15-16).


.. py:function:: writhe_method_1b(points: numpy.typing.ArrayLike, *, closed: bool = True, eps: float = 1e-12) -> tuple[float, float]

   Method 1b: analytic Gauss integral (Eqs 13, 24-25).
   Returns (writhe, acn_like).


.. py:function:: writhe_method_2a(points: numpy.typing.ArrayLike, *, eps: float = 1e-12) -> float

   Method 2a: Wr = Twz + Wrz - Tw (Eqs 30-34), z-axis projection. Requires a
   closed chain.


.. py:function:: writhe_method_2b(points: numpy.typing.ArrayLike, *, eps: float = 1e-12) -> float

   Method 2b (le Bret-style):
       Wr = Wrz - Tw with a_i = k×s_i/|k×s_i| (Eqs 35-38), k = z-axis.
   Requires CLOSED chain.


