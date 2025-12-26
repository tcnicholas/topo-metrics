Periodic nets and quotient graphs
=================================

Crystal structures are often idealised as **nets**: periodic graphs in
which vertices represent sites and edges represent connections between
them. Here we formalise that idea and introduce the vector representation
and quotient graph.


Periodic nets
-------------

A **net** in the sense used here is:

* a **connected**, **simple** graph (no loops, no multiple edges),
* realised as a **periodic** structure in Euclidean space.

We say that a net is **n-periodic** if it has translational symmetry in
exactly ``n`` linearly independent directions (for crystal structures of
interest, ``n = 3``). The repeat unit is typically taken to be a
primitive unit cell.

It is important to distinguish:

* **n-periodic**: the net has ``n`` independent translation vectors
  (e.g. a 3-periodic net fills 3D space with translational symmetry),
* **n-dimensional**: the net admits a faithful embedding in ``n``
  dimensions but not in ``n-1`` dimensions.

A graph can be 2-dimensional (planar) and still be 3-periodic if it is
embedded periodically in three-dimensional space, e.g. a layered
material.


Vector representation of a periodic net
---------------------------------------

For a periodic net, it is convenient to choose:

* a **primitive unit cell** with basis vectors :math:`\mathbf{a},
  \mathbf{b}, \mathbf{c}`, and
* a numbering of the vertices in one reference cell:
  :math:`1, 2, \dots, v`.

Every edge in the infinite periodic net can be represented as a tuple

.. math::

   (i, j, u, v, w),

meaning:

* the edge connects vertex ``i`` in the reference cell to
* vertex ``j`` in the cell shifted by the lattice vector
  :math:`u \mathbf{a} + v \mathbf{b} + w \mathbf{c}`.

The list of all such edge tuples within the primitive cell is called the
**vector representation** of the net. For simple graphs we only store one
tuple for each undirected edge.

This representation plays, for periodic nets, the same role that an
adjacency matrix plays for finite graphs. In principle, all topological
properties (ring sizes, connectivity, combinatorial symmetry, etc.) can
be recovered from it.


Quotient graph
--------------

The **quotient graph** compresses the infinite periodic net down to a
finite labelled graph associated with a chosen unit cell:

* The vertices of the quotient graph correspond to vertices in the
  reference cell.
* Each edge of the quotient graph corresponds to an edge tuple
  :math:`(i, j, u, v, w)` in the vector representation.
* The edge is labelled by the translation :math:`(u, v, w)`.

For example, in the diamond net (dia) there are two vertices in the
primitive cell (labelled 1 and 2) and four edges in the vector
representation:

* (1, 2, 0, 0, 0)
* (1, 2, 1, 0, 0)
* (1, 2, 0, 1, 0)
* (1, 2, 0, 0, 1)

The quotient graph therefore has two vertices, with four edges between
them carrying these labels. It is allowed to have loops and multiple
edges.


Genus and minimal nets
----------------------

The **cyclomatic number** of the quotient graph is sometimes called the
**genus** of the net. For a connected quotient graph with ``e`` edges
and ``v`` vertices this is

.. math::

   g = 1 + e - v.

The minimum possible genus of an n-periodic net is ``n``. Nets that
achieve this minimum are called **minimal nets**. For example, a
3-periodic net must have genus at least 3; those with genus 3 are minimal
in this sense.

Minimal nets and quotient graphs are useful both for classification and
for constructing canonical representations of periodic nets, which in
turn allows algorithms to test when two periodic nets are topologically
equivalent.
