Vertex symbols and face symbols
===============================

Local environments in periodic nets are often summarised by compact
symbols that encode the sizes of rings around each vertex or the faces
around each tile. Here we outline the most common conventions:
point/Schläfli symbols, vertex symbols, and face symbols.


Point (Schläfli) symbols
------------------------

For a vertex of coordination number ``n`` there are
:math:`n (n - 1) / 2` angles (unordered pairs of incident edges).

The **point symbol** (also called a Schläfli symbol in this context) of
a vertex collects information about the sizes of the *shortest* cycles
passing through each angle. It is written as

.. math::

   A^{a} . B^{b} . C^{c} . \dots

where:

* :math:`A, B, C, \dots` are ring sizes (e.g. 4, 6, 8),
* :math:`a, b, c, \dots` tell you how many angles see a shortest ring
  of that size,
* and :math:`a + b + c + \dots = n (n - 1) / 2`.

Examples:

* In the 4-coordinated diamond net (dia) the shortest cycles through any
  angle are all 6-rings, so the point symbol is simply ``6^6``.
* In the primitive cubic lattice (pcu, coordination 6) the point symbol
  is ``4^12 6^3``: twelve angles see a smallest 4-ring, three see a
  smallest 6-ring.


Vertex symbols (long symbols)
-----------------------------

A more detailed **vertex symbol** lists the sizes of shortest rings
through each angle separately. For a 3-coordinated vertex one writes
something like

.. math::

   A_{x} \; B_{y} \; C_{z},

meaning that:

* the first angle is part of ``x`` different shortest ``A``-rings,
* the second angle is part of ``y`` different shortest ``B``-rings,
* the third angle is part of ``z`` different shortest ``C``-rings.

Example: in the 3-coordinated net ``srs`` (Si sublattice of SrSi\ :sub:`2`)
there are five 10-rings at each angle, so the vertex symbol is

.. math::

   10_5 \; 10_5 \; 10_5.

For the 4-coordinated diamond net (dia) there are six angles, and at
each angle the shortest rings are 6-rings, so a vertex symbol is

.. math::

   6_2 \; 6_2 \; 6_2 \; 6_2 \; 6_2 \; 6_2,

indicating that each angle lies on two distinct 6-rings.

By convention, angles in 4-coordinated nets are grouped into three pairs
of opposite angles, and vertex symbols are ordered with smaller ring
sizes first within this grouping.

If a particular angle does *not* lie on any ring (in the relevant
sense), an asterisk ``*`` is often used as a placeholder for that angle.