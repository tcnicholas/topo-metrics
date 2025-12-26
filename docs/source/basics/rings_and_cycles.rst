Rings and cycles in crystal nets
================================

Rings are one of the most important descriptors of crystal-net topology.
This page clarifies what we mean by cycles, rings, and strong rings, and
how they relate to other notions such as ring sums and smallest sets of
rings.


Cycles vs rings
---------------

Recall that an **elementary cycle** is a closed path that does not visit
any vertex or edge more than once (apart from the start/end vertex).

Different communities use "ring" in slightly different ways:

* In some molecular graph conventions, **every** elementary cycle is
  called a ring, and the **cyclomatic number** of the graph is the
  "ring count" (e.g. cubane is pentacyclic).
* In solid-state crystal chemistry it is more common to reserve "ring"
  for a more specific kind of cycle, tied to geometric shortest paths.


Ring sums
---------

Given two cycles, their **ring sum** is defined as follows:

* Consider the set of edges that belong to exactly one of the cycles
  (symmetric difference of the edge sets).
* That set of edges either forms one or more cycles, or possibly
  disconnects.

This operation extends to any finite set of cycles: take all edges that
appear an odd number of times across the set.

The ring sum is useful for reasoning about when one cycle can be built
as a combination of others.


Rings in the solid-state sense
------------------------------

In the solid-state context, a **ring** is usually defined as a cycle
with the property that:

* there is no strictly shorter path ("shortcut") between any two vertices
  of the cycle than the path along the cycle itself.

An equivalent, purely graph-theoretic formulation is:

* a ring is an elementary cycle that cannot be written as the ring sum
  of two *shorter* cycles.

We speak of an **n-ring** for a ring that contains ``n`` edges (or
equivalently, ``n`` vertices).

This definition excludes cycles that are "composite" in the sense that
they can be built by combining smaller cycles. For example, in the graph
of a cube there are:

* six square 4-rings (the faces),
* 6-cycles that are sums of three 4-rings,
* 8-cycles that are sums of several smaller cycles.

Only some of these are rings in the above sense; others are considered
derivative.


Strong rings
------------

Goetzke and Klein introduced the useful notion of a **strong ring**, a
ring that cannot be written as the ring sum of any collection of
*smaller* cycles, not just two. Every strong ring is a ring, but the converse 
need not be true.