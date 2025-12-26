Graphs and crystal nets
=======================

This page fixes basic graph-theoretic terminology as it is used in
crystal-net topology and throughout TopoMetrics. The conventions here
follow the review of crystal nets as graphs by Delgado-Friedrichs and
O'Keeffe.


Vertices, edges, and neighbors
------------------------------

A **graph** consists of a set of vertices (nodes) and a set of edges
(pairs of vertices):

* vertices typically correspond to atomic sites (or clusters of atoms),
* edges correspond to some notion of connection (e.g., bonds or neighbours).

Unless stated otherwise, we work with **undirected** graphs: an edge
between vertices ``i`` and ``j`` does not carry an orientation.

We usually exclude loops and multiple edges:

* A **loop** is an edge from a vertex to itself.
* **Multiple edges** are several distinct edges between the same two
  vertices.

A graph with no loops and no multiple edges is called a **simple graph**.
Crystal nets are normally treated as simple graphs.

If two vertices are connected by an edge they are **neighbours**, and the
set of all neighbors of a vertex is its **neighbourhood**. The
**coordination number** of a vertex is the number of incident edges
(i.e., the size of its neighborhood). In graph theory this is also called
the vertex *degree*, but "degree" and "valence" already have meanings in
chemistry, so we will prefer **coordination number** here.

A graph in which every vertex has the same coordination number ``n`` is
called **n-regular** (or simply "regular").


Paths, cycles, and connectivity
-------------------------------

A **path** (also called a *chain*) is a sequence of vertices

.. math::

   (x_1, x_2, \dots, x_n)

such that each consecutive pair :math:`(x_k, x_{k+1})` is joined by an
edge. The length of the path is the number of edges it contains.

A **cycle** (or **circuit**) is a closed path that starts and ends at
the same vertex:

.. math::

   (x_1, x_2, \dots, x_{n-1}, x_1).

We are usually interested in **elementary** cycles: cycles in which no
edge or vertex (other than the start/end) is repeated.

A graph is **connected** if there is at least one path between every
pair of vertices.

Removing a vertex (and all incident edges) is called **deletion** of
that vertex. A graph is called **n-connected** if it has at least
``n + 1`` vertices and deleting any set of fewer than ``n`` vertices
never disconnects it.


Embeddings and planarity
------------------------

A **graph** is a combinatorial object: it does not come with coordinates
for its vertices. Choosing coordinates and drawing edges as straight
segments in Euclidean space is called an **embedding** (or
**realisation**) of the graph.

An embedding is **faithful** if edges only meet at their endpoints: they
do not cross and do not pass through other vertices.

* A graph is **planar** if it admits a faithful embedding in the plane
  (two-dimensional space).
* Any finite graph can be embedded faithfully in three-dimensional
  space.


Graph isomorphism and symmetry
-------------------------------

Two graphs are **isomorphic** if there is a one-to-one correspondence
between their vertices that preserves adjacency (edges). Intuitively,
they are the same graph, drawn differently.

An **automorphism** of a graph is an isomorphism from the graph to
itself. The set of all automorphisms forms the **graph group**, which is
a purely combinatorial notion of symmetry.
