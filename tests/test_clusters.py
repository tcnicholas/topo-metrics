import numpy as np
import pytest
from pymatgen.core import Lattice as PymatgenLattice

from topo_metrics.clusters import (
    Cluster,
    get_carvs_std_dev,
    get_carvs_vector,
    get_clusters,
    get_opposite_angles,
    get_ring_size_by_angle,
    get_ring_sizes,
    get_vertex_symbol,
    largest_ring_size,
    number_of_angles,
    number_of_rings,
    only_smallest_ring_size,
    smallest_ring_size,
)
from topo_metrics.rings import Ring
from topo_metrics.symbols import VertexSymbol
from topo_metrics.topology import Node


@pytest.fixture
def sample_lattice():
    return PymatgenLattice.cubic(5.0)


@pytest.fixture
def sample_cluster(sample_lattice):
    """Create a simple cluster with two rings."""

    nodes = {
        i: Node(node_id=i, frac_coord=np.random.rand(3)) for i in range(1, 7)
    }

    def make_ring(node_ids):
        ring_nodes = [nodes[i] for i in node_ids]
        angle = ((node_ids[-1], (0, 0, 0)), (node_ids[1], (0, 0, 0)))
        return Ring(nodes=ring_nodes, angle=angle)

    # create a 4-ring and a 5-ring
    ring4 = make_ring([1, 2, 3, 4])
    ring5 = make_ring([1, 5, 6, 3, 2])

    return Cluster(central_node_id=1, rings=[ring4, ring5])


def test_ring_properties(sample_cluster):
    """Test the properties of a ring."""

    assert smallest_ring_size(sample_cluster) == 4
    assert largest_ring_size(sample_cluster) == 5
    assert number_of_rings(sample_cluster) == 2
    assert number_of_angles(sample_cluster) == 2


def test_get_ring_sizes(sample_cluster):
    """Test the get_ring_sizes function."""

    assert get_ring_sizes(sample_cluster) == [4, 5]


########## TEST HELPERS ##########


def test_get_opposite_angles_finds_pairs():
    """Test that get_opposite_angles finds pairs of angles that share a node."""

    angles = [
        ((4, (-1, 0, -1)), (2, (0, 0, 0))),
        ((4, (0, 0, -1)), (2, (0, -1, 0))),
        ((2, (0, 0, 0)), (2, (0, -1, 0))),
        ((4, (-1, 0, -1)), (2, (0, -1, 0))),
        ((4, (0, 0, -1)), (2, (0, 0, 0))),
        ((4, (0, 0, -1)), (4, (-1, 0, -1))),
    ]

    expected = [
        (
            ((4, (-1, 0, -1)), (2, (0, 0, 0))),
            ((4, (0, 0, -1)), (2, (0, -1, 0))),
        ),
        (
            ((2, (0, 0, 0)), (2, (0, -1, 0))),
            ((4, (0, 0, -1)), (4, (-1, 0, -1))),
        ),
        (
            ((4, (-1, 0, -1)), (2, (0, -1, 0))),
            ((4, (0, 0, -1)), (2, (0, 0, 0))),
        ),
    ]

    result = get_opposite_angles(angles)

    assert result == expected


def test_get_opposite_angles_no_pairs():
    """Ensure inner loop runs even when no pairs are found."""

    angles = [
        ((1, (0, 0, 0)), (2, (0, 0, 0))),
        ((2, (0, 0, 0)), (3, (0, 0, 0))),  # shares node with first
        ((3, (0, 0, 0)), (1, (0, 0, 0))),  # shares node with both above
    ]

    result = get_opposite_angles(angles)

    assert result == []


def test_only_smallest_ring_size_multiple_min():
    """Test that multiple smallest sizes are all retained."""

    ring_sizes = [4, 5, 4, 6]
    result = only_smallest_ring_size(ring_sizes)

    assert result == [4, 4]


def test_only_smallest_ring_size_single_min():
    """Test with all different values."""

    ring_sizes = [6, 3, 5, 4]
    result = only_smallest_ring_size(ring_sizes)

    assert result == [3]


def test_only_smallest_ring_size_all_same():
    """Test with all elements equal."""

    ring_sizes = [5, 5, 5]
    result = only_smallest_ring_size(ring_sizes)

    assert result == [5, 5, 5]


def test_only_smallest_ring_size_empty_list():
    """Test empty input list."""

    result = only_smallest_ring_size([])
    assert result == []


########## TEST ALPHA-CRISTOBALITE ##########


def test_get_clusters(sample_topology, sample_converted_rnodes):
    """Test the get_clusters function."""

    clusters = get_clusters(sample_topology, sample_converted_rnodes)

    assert len(clusters) == 4
    assert all(isinstance(cluster, Cluster) for cluster in clusters)


def test_ring_size_by_angle(sample_topology, sample_converted_rnodes):
    """Test that the ring size is determined by the angle."""

    clusters = get_clusters(sample_topology, sample_converted_rnodes)
    eg_cluster = clusters[0]

    result = get_ring_size_by_angle(eg_cluster, all_rings=True)

    expected = {
        ((4, (0, 0, -1)), (4, (-1, 0, -1))): [6, 6],
        ((4, (0, 0, -1)), (2, (0, -1, 0))): [6, 6],
        ((4, (-1, 0, -1)), (2, (0, -1, 0))): [6, 6],
        ((2, (0, 0, 0)), (2, (0, -1, 0))): [6, 6],
        ((4, (0, 0, -1)), (2, (0, 0, 0))): [6, 6],
        ((4, (-1, 0, -1)), (2, (0, 0, 0))): [6, 6],
    }

    assert result == expected

    # the same result is expected when all_rings is False for this cluster.
    result2 = get_ring_size_by_angle(eg_cluster, all_rings=False)

    assert result2 == expected


########## TEST VERTEX SYMBOLS ##########


def test_get_vertex_symbol(sample_topology, sample_converted_rnodes):
    """Test that the vertex symbol is determined correctly."""

    clusters = get_clusters(sample_topology, sample_converted_rnodes)
    eg_cluster = clusters[0]

    result = get_vertex_symbol(eg_cluster)

    expected = VertexSymbol(
        vector=[[6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6]],
        vector_all_rings=[[6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6]],
    )

    assert result == expected

    result2 = get_vertex_symbol(clusters)

    assert result2 == [expected] * 4


def test_vertex_symbol_tie_breaks_with_all_rings(sample_lattice):
    """Test that tied opposite angles are sorted using all_rings vector."""

    # Create 6 nodes
    nodes = {
        i: Node(node_id=i, frac_coord=np.random.rand(3)) for i in range(1, 7)
    }

    def make_ring(node_ids, angle):
        ring_nodes = [nodes[i] for i in node_ids]
        return Ring(nodes=ring_nodes, angle=angle)

    # define two disjoint angles = opposite.
    angle_a = ((1, (0, 0, 0)), (2, (0, 0, 0)))
    angle_b = ((3, (0, 0, 0)), (4, (0, 0, 0)))

    # angle A: two rings of size 6
    ring_a1 = make_ring([1, 2, 5, 6, 1, 2], angle_a)
    ring_a2 = make_ring([2, 1, 6, 5, 2, 1], angle_a)

    # angle B: two rings of size 6, one additional smaller ring.
    ring_b1 = make_ring([3, 4, 5, 6, 3, 4], angle_b)
    ring_b2 = make_ring([4, 3, 6, 5, 4, 3], angle_b)
    ring_b3 = make_ring([3, 4, 5, 3], angle_b)

    cluster = Cluster(
        central_node_id=1, rings=[ring_a1, ring_a2, ring_b1, ring_b2, ring_b3]
    )

    result = get_vertex_symbol(cluster)

    assert result.vector == [[4], [6, 6]]
    assert result.vector_all_rings == [[4, 6, 6], [6, 6]]


def test_vertex_symbol_tie_breaks_4_connected_net(sample_lattice):
    """Test tie-breaking occurs correctly for 4-connected nets."""

    # 1 central node + 4 neighbours
    nodes = {
        i: Node(node_id=i, frac_coord=np.random.rand(3)) for i in range(1, 6)
    }

    def make_ring(node_ids, angle):
        return Ring(nodes=[nodes[i] for i in node_ids], angle=angle)

    # central node is 1, neighbours are 2, 3, 4, 5
    # angles: (2,3), (2,4), (2,5), (3,4), (3,5), (4,5)
    angles = [
        ((2, (0, 0, 0)), (3, (0, 0, 0))),
        ((2, (0, 0, 0)), (4, (0, 0, 0))),
        ((2, (0, 0, 0)), (5, (0, 0, 0))),
        ((3, (0, 0, 0)), (4, (0, 0, 0))),
        ((3, (0, 0, 0)), (5, (0, 0, 0))),
        ((4, (0, 0, 0)), (5, (0, 0, 0))),
    ]

    rings = []

    # Add one ring of size 6 to each angle
    for angle in angles:
        nid1, nid2 = angle[0][0], angle[1][0]
        rings.append(make_ring([1, nid1, nid2, 1, nid1, nid2], angle))  # size 6

    # Choose two opposite angles — e.g., (2,3) and (4,5)
    angle_a = angles[0]  # (2,3)
    angle_b = angles[5]  # (4,5)

    # Add an *extra* ring of same size to angle_a (→ smallest ring size still 6)
    rings.append(make_ring([1, 2, 3, 1, 2, 3], angle_a))  # size 6 again

    # Add an *extra* smaller ring to angle_b (→ smallest becomes 4)
    rings.append(make_ring([1, 4, 5, 1], angle_b))  # size 3 or 4

    cluster = Cluster(central_node_id=1, rings=rings)

    result = get_vertex_symbol(cluster)

    assert result.vector == [[4], [6, 6], [6], [6], [6], [6]]
    assert result.vector_all_rings == [[4, 6], [6, 6], [6], [6], [6], [6]]


########## TEST CARVS ##########


@pytest.mark.parametrize(
    "max_size, expected_vector",
    [
        (None, np.array([0, 0, 0, 0, 0, 12])),  # auto-detect max size (6)
        (6, np.array([0, 0, 0, 0, 0, 12])),  # explicitly set max size to 6
        (8, np.array([0, 0, 0, 0, 0, 12, 0, 0])),  # padded with zeros
    ],
)
def test_get_carvs_vector(
    sample_topology, sample_converted_rnodes, max_size, expected_vector
):
    """Test CARVS vector is determined correctly for various max_size values."""

    clusters = get_clusters(sample_topology, sample_converted_rnodes)

    # test single cluster.
    result = get_carvs_vector(clusters[0], max_size=max_size)

    np.testing.assert_array_equal(result.vector, expected_vector)
    assert result.spread == 0.0
    assert result.is_single_node is True

    # test full cluster list.
    result2 = get_carvs_vector(clusters, max_size=max_size)

    np.testing.assert_array_equal(result2.vector, expected_vector)
    assert result2.spread == 0.0
    assert result2.is_single_node is False


def test_get_carvs_vector_per_clusters(
    sample_topology, sample_converted_rnodes
):
    """Test that the CARVS vector is determined correctly for each cluster."""

    clusters = get_clusters(sample_topology, sample_converted_rnodes)

    result = get_carvs_vector(clusters, return_per_cluster=True)

    expected = np.array(
        [
            [0, 0, 0, 0, 0, 12],
            [0, 0, 0, 0, 0, 12],
            [0, 0, 0, 0, 0, 12],
            [0, 0, 0, 0, 0, 12],
        ]
    )

    np.testing.assert_array_equal(result, expected)


def test_get_carvs_vector_raises_on_invalid_input():
    """Test that get_carvs_vector raises TypeError on invalid input."""

    invalid_input = "not a cluster"

    with pytest.raises(
        TypeError, match="Expected a Cluster or list of Cluster objects"
    ):
        get_carvs_vector(invalid_input)


def test_get_carvs_vector_per_clusters_unqiue(
    sample_topology, sample_converted_rnodes
):
    """Test that the unique CARVS vector is determined correctly."""

    clusters = get_clusters(sample_topology, sample_converted_rnodes)

    result = get_carvs_vector(clusters, return_per_cluster=True, unique=True)

    expected = np.array([[0, 0, 0, 0, 0, 12]])

    np.testing.assert_array_equal(result, expected)


def test_get_carvs_vector_warns_on_small_max_size(
    sample_topology, sample_converted_rnodes
):
    """Test warning is raised when max_size is smaller than required."""

    clusters = get_clusters(sample_topology, sample_converted_rnodes)

    too_small_max_size = 4  # actual max ring size is 6 in your data

    with pytest.warns(
        UserWarning, match="The maximum ring size in the network is 6"
    ):
        result = get_carvs_vector(clusters, max_size=too_small_max_size)

    # Check that it still returns a correct vector with the adjusted max size
    expected_vector = np.array([0, 0, 0, 0, 0, 12])
    np.testing.assert_array_equal(result.vector, expected_vector)
    assert result.spread == 0.0
    assert result.is_single_node is False


def test_get_carvs_std_dev_single_vector():
    """Test that standard deviation is 0.0 when given a single CARVS vector."""

    vector = np.array([0, 0, 0, 0, 0, 12])
    result = get_carvs_std_dev(vector)

    assert result == 0.0


def test_shares_edge():
    """Test the shares_edge function."""
    from topo_metrics.clusters import shares_edge

    # Angles that share an edge
    angle1 = ((1, (0, 0, 0)), (2, (0, 0, 0)))
    angle2 = ((2, (0, 0, 0)), (3, (0, 0, 0)))
    assert shares_edge(angle1, angle2) is True

    # Angles that don't share an edge
    angle3 = ((4, (0, 0, 0)), (5, (0, 0, 0)))
    assert shares_edge(angle1, angle3) is False

    # Same angle
    assert shares_edge(angle1, angle1) is True


def test_get_unique_angles(sample_cluster):
    """Test getting unique angles from a cluster."""
    from topo_metrics.clusters import get_unique_angles

    unique_angles = get_unique_angles(sample_cluster)
    assert isinstance(unique_angles, list)
    # Should have unique angles
    assert len(unique_angles) == len(set(unique_angles))


def test_largest_ring_size_empty():
    """Test largest_ring_size with empty cluster."""

    empty_cluster = Cluster(central_node_id=1, rings=[])
    assert largest_ring_size(empty_cluster) == 0


def test_smallest_and_largest_ring_size(sample_cluster):
    """Test smallest and largest ring sizes."""

    # Sample cluster has a 4-ring and 5-ring
    assert smallest_ring_size(sample_cluster) == 4
    assert largest_ring_size(sample_cluster) == 5


def test_cluster_repr(sample_cluster):
    """Test the __repr__ method of Cluster."""

    repr_str = repr(sample_cluster)
    assert "Cluster" in repr_str
    assert "node_id" in repr_str
    assert "n_rings" in repr_str
    assert "CARVS" in repr_str


def test_get_carvs_vector_empty_cluster():
    """Test get_carvs_vector with an empty cluster (no rings)."""

    empty_cluster = Cluster(central_node_id=1, rings=[])
    result = get_carvs_vector(empty_cluster)

    # Should return a CARVS with vector of zeros
    assert result.is_single_node is True
    assert result.spread == 0.0
    assert len(result.vector) == 1
    assert result.vector[0] == 0
