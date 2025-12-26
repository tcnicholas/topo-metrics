topo_metrics.ring_geometry
==========================

.. py:module:: topo_metrics.ring_geometry


Classes
-------

.. autoapisummary::

   topo_metrics.ring_geometry.RingGeometry


Module Contents
---------------

.. py:class:: RingGeometry

   .. py:attribute:: nodes
      :type:  tuple[topo_metrics.topology.Node, Ellipsis]

      The nodes in the ring.


   .. py:property:: species
      :type: str


      Species string of the ring.


   .. py:property:: positions
      :type: numpy.typing.NDArray[numpy.floating]


      Cartesian positions of the nodes in the ring.


   .. py:property:: radius_of_gyration
      :type: float


      Radius of gyration around the geometric centroid.


   .. py:property:: gyration_tensor
      :type: numpy.typing.NDArray[numpy.floating]


      Gyration tensor of the ring.

      The gyration tensor describes the second moments of position of a set of
      points around their center of mass. It is a symmetric 3x3 matrix.


   .. py:property:: principal_moments
      :type: numpy.typing.NDArray[numpy.floating]


      Principal moments of the gyration tensor.

      The principal moments are the eigenvalues of the gyration tensor, which
      describe the distribution of points along the principal axes.


   .. py:property:: asphericity
      :type: float


      Asphericity of the ring based on the principal moments.


   .. py:method:: writhe_and_acn(method: str = '1a', closed=True) -> tuple[float, float] | float

      Writhe of the ring using specified method from

      :param method: Method to compute writhe. Options are '1a', '1b', '2a', '2b'.
                     Default is '1a'. Each method corresponds to those introduced in

      :rtype: Writhe of the ring.



   .. py:property:: geometric_centroid
      :type: numpy.typing.NDArray[numpy.floating]


      Geometric centroid of the ring.


   .. py:method:: to_xyz(filename: pathlib.Path | str, write_info: bool = False) -> None

      Write the ring to an xyz file.



