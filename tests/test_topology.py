from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from topo_metrics.rings import RingSizeCounts
from topo_metrics.topology import Node, RingsResults, Topology


def test_topology_getitem_valid():
    """Test that __getitem__ returns the correct node."""

    t = Topology(nodes=[Node(1, "X"), Node(2, "Y")])

    assert t[1].node_type == "X"


def test_topology_getitem_invalid():
    """Test that __getitem__ raises an error for an invalid node_id."""

    t = Topology(nodes=[Node(1, "X")])

    with pytest.raises(IndexError, match="out of bounds"):
        _ = t[5]


def test_topology_from_cgd_file_not_found():
    """Test that an exception is raised when the CGD file is not found."""

    with patch("os.path.exists", return_value=False), pytest.raises(
        FileNotFoundError, match="nonexistent.cgd"
    ):
        Topology.from_cgd("nonexistent.cgd")


def test_topology_from_cgd(dia_file: Path):
    """Test that Topology can be created from a CGD file."""

    topology = Topology.from_cgd(str(dia_file))

    assert len(topology.nodes) == 4
    assert topology.edges.shape == (16, 5)
    assert topology.lattice is not None
    assert isinstance(topology[1], Node)


def patched_run_rings(edges, depth):
    rcount = np.array(
        [
            [1, 0],
            [2, 0],
            [3, 0],
            [4, 0],
            [5, 0],
            [6, 8],
            [7, 0],
            [8, 0],
            [9, 0],
            [10, 0],
            [11, 0],
            [12, 0],
            [13, 0],
            [14, 0],
            [15, 0],
            [16, 0],
            [17, 0],
            [18, 0],
            [19, 0],
            [20, 0],
            [21, 0],
            [22, 0],
            [23, 0],
            [24, 0],
            [25, 0],
            [26, 0],
            [27, 0],
        ]
    )
    rings_per_node = np.array([12, 12, 12, 12])
    rnodes = [
        [
            [
                (1, [0, 0, 0]),
                (4, [-1, 0, -1]),
                (3, [-1, -1, -1]),
                (2, [0, -1, -1]),
                (3, [0, -1, -1]),
                (4, [0, 0, -1]),
            ],
            [
                (1, [0, 0, 0]),
                (4, [-1, 0, -1]),
                (3, [-1, 0, -1]),
                (2, [0, 0, -1]),
                (3, [0, 0, -1]),
                (4, [0, 0, -1]),
            ],
            [
                (1, [0, 0, 0]),
                (2, [0, -1, 0]),
                (3, [0, -1, 0]),
                (2, [1, -1, 0]),
                (1, [1, 0, 0]),
                (4, [0, 0, -1]),
            ],
            [
                (1, [-1, 0, 0]),
                (2, [-1, -1, 0]),
                (3, [-1, -1, 0]),
                (2, [0, -1, 0]),
                (1, [0, 0, 0]),
                (4, [-1, 0, -1]),
            ],
            [
                (1, [0, 0, 0]),
                (2, [0, -1, 0]),
                (3, [0, -1, 0]),
                (4, [0, 0, 0]),
                (3, [0, 0, 0]),
                (2, [0, 0, 0]),
            ],
            [
                (1, [0, 0, 0]),
                (2, [0, -1, 0]),
                (3, [-1, -1, 0]),
                (4, [-1, 0, 0]),
                (3, [-1, 0, 0]),
                (2, [0, 0, 0]),
            ],
            [
                (1, [0, 0, 0]),
                (2, [0, 0, 0]),
                (1, [0, 1, 0]),
                (4, [0, 1, -1]),
                (3, [0, 0, -1]),
                (4, [0, 0, -1]),
            ],
            [
                (1, [0, -1, 0]),
                (2, [0, -1, 0]),
                (1, [0, 0, 0]),
                (4, [0, 0, -1]),
                (3, [0, -1, -1]),
                (4, [0, -1, -1]),
            ],
            [
                (1, [0, 0, 0]),
                (2, [0, 0, 0]),
                (3, [0, 0, 0]),
                (2, [1, 0, 0]),
                (1, [1, 0, 0]),
                (4, [0, 0, -1]),
            ],
            [
                (1, [-1, 0, 0]),
                (2, [-1, 0, 0]),
                (3, [-1, 0, 0]),
                (2, [0, 0, 0]),
                (1, [0, 0, 0]),
                (4, [-1, 0, -1]),
            ],
            [
                (1, [0, 0, 0]),
                (2, [0, 0, 0]),
                (1, [0, 1, 0]),
                (4, [-1, 1, -1]),
                (3, [-1, 0, -1]),
                (4, [-1, 0, -1]),
            ],
            [
                (1, [0, -1, 0]),
                (2, [0, -1, 0]),
                (1, [0, 0, 0]),
                (4, [-1, 0, -1]),
                (3, [-1, -1, -1]),
                (4, [-1, -1, -1]),
            ],
        ],
        [
            [
                (1, [0, 1, 1]),
                (4, [-1, 1, 0]),
                (3, [-1, 0, 0]),
                (2, [0, 0, 0]),
                (3, [0, 0, 0]),
                (4, [0, 1, 0]),
            ],
            [
                (1, [0, 0, 1]),
                (4, [-1, 0, 0]),
                (3, [-1, 0, 0]),
                (2, [0, 0, 0]),
                (3, [0, 0, 0]),
                (4, [0, 0, 0]),
            ],
            [
                (1, [0, 1, 0]),
                (2, [0, 0, 0]),
                (3, [0, 0, 0]),
                (2, [1, 0, 0]),
                (1, [1, 1, 0]),
                (4, [0, 1, -1]),
            ],
            [
                (1, [-1, 1, 0]),
                (2, [-1, 0, 0]),
                (3, [-1, 0, 0]),
                (2, [0, 0, 0]),
                (1, [0, 1, 0]),
                (4, [-1, 1, -1]),
            ],
            [
                (1, [0, 1, 0]),
                (2, [0, 0, 0]),
                (3, [0, 0, 0]),
                (4, [0, 1, 0]),
                (3, [0, 1, 0]),
                (2, [0, 1, 0]),
            ],
            [
                (1, [0, 0, 0]),
                (2, [0, -1, 0]),
                (3, [0, -1, 0]),
                (4, [0, 0, 0]),
                (3, [0, 0, 0]),
                (2, [0, 0, 0]),
            ],
            [
                (1, [0, 1, 0]),
                (2, [0, 0, 0]),
                (3, [-1, 0, 0]),
                (4, [-1, 1, 0]),
                (3, [-1, 1, 0]),
                (2, [0, 1, 0]),
            ],
            [
                (1, [0, 0, 0]),
                (2, [0, -1, 0]),
                (3, [-1, -1, 0]),
                (4, [-1, 0, 0]),
                (3, [-1, 0, 0]),
                (2, [0, 0, 0]),
            ],
            [
                (1, [0, 0, 0]),
                (2, [0, 0, 0]),
                (1, [0, 1, 0]),
                (4, [0, 1, -1]),
                (3, [0, 0, -1]),
                (4, [0, 0, -1]),
            ],
            [
                (1, [0, 0, 0]),
                (2, [0, 0, 0]),
                (3, [0, 0, 0]),
                (2, [1, 0, 0]),
                (1, [1, 0, 0]),
                (4, [0, 0, -1]),
            ],
            [
                (1, [-1, 0, 0]),
                (2, [-1, 0, 0]),
                (3, [-1, 0, 0]),
                (2, [0, 0, 0]),
                (1, [0, 0, 0]),
                (4, [-1, 0, -1]),
            ],
            [
                (1, [0, 0, 0]),
                (2, [0, 0, 0]),
                (1, [0, 1, 0]),
                (4, [-1, 1, -1]),
                (3, [-1, 0, -1]),
                (4, [-1, 0, -1]),
            ],
        ],
        [
            [
                (1, [1, 1, 1]),
                (4, [0, 1, 0]),
                (3, [0, 0, 0]),
                (2, [1, 0, 0]),
                (3, [1, 0, 0]),
                (4, [1, 1, 0]),
            ],
            [
                (1, [0, 1, 1]),
                (4, [-1, 1, 0]),
                (3, [-1, 0, 0]),
                (2, [0, 0, 0]),
                (3, [0, 0, 0]),
                (4, [0, 1, 0]),
            ],
            [
                (1, [1, 0, 1]),
                (4, [0, 0, 0]),
                (3, [0, 0, 0]),
                (2, [1, 0, 0]),
                (3, [1, 0, 0]),
                (4, [1, 0, 0]),
            ],
            [
                (1, [0, 0, 1]),
                (4, [-1, 0, 0]),
                (3, [-1, 0, 0]),
                (2, [0, 0, 0]),
                (3, [0, 0, 0]),
                (4, [0, 0, 0]),
            ],
            [
                (1, [0, 1, 0]),
                (2, [0, 0, 0]),
                (3, [0, 0, 0]),
                (2, [1, 0, 0]),
                (1, [1, 1, 0]),
                (4, [0, 1, -1]),
            ],
            [
                (1, [0, 1, 0]),
                (2, [0, 0, 0]),
                (3, [0, 0, 0]),
                (4, [0, 1, 0]),
                (3, [0, 1, 0]),
                (2, [0, 1, 0]),
            ],
            [
                (1, [0, 0, 0]),
                (2, [0, -1, 0]),
                (3, [0, -1, 0]),
                (4, [0, 0, 0]),
                (3, [0, 0, 0]),
                (2, [0, 0, 0]),
            ],
            [
                (1, [1, 1, 0]),
                (2, [1, 0, 0]),
                (3, [0, 0, 0]),
                (4, [0, 1, 0]),
                (3, [0, 1, 0]),
                (2, [1, 1, 0]),
            ],
            [
                (1, [1, 0, 0]),
                (2, [1, -1, 0]),
                (3, [0, -1, 0]),
                (4, [0, 0, 0]),
                (3, [0, 0, 0]),
                (2, [1, 0, 0]),
            ],
            [
                (1, [0, 0, 1]),
                (2, [0, 0, 1]),
                (1, [0, 1, 1]),
                (4, [0, 1, 0]),
                (3, [0, 0, 0]),
                (4, [0, 0, 0]),
            ],
            [
                (1, [0, 0, 0]),
                (2, [0, 0, 0]),
                (3, [0, 0, 0]),
                (2, [1, 0, 0]),
                (1, [1, 0, 0]),
                (4, [0, 0, -1]),
            ],
            [
                (1, [1, 0, 1]),
                (2, [1, 0, 1]),
                (1, [1, 1, 1]),
                (4, [0, 1, 0]),
                (3, [0, 0, 0]),
                (4, [0, 0, 0]),
            ],
        ],
        [
            [
                (1, [1, 0, 1]),
                (4, [0, 0, 0]),
                (3, [0, -1, 0]),
                (2, [1, -1, 0]),
                (3, [1, -1, 0]),
                (4, [1, 0, 0]),
            ],
            [
                (1, [0, 0, 1]),
                (4, [-1, 0, 0]),
                (3, [-1, -1, 0]),
                (2, [0, -1, 0]),
                (3, [0, -1, 0]),
                (4, [0, 0, 0]),
            ],
            [
                (1, [1, 0, 1]),
                (4, [0, 0, 0]),
                (3, [0, 0, 0]),
                (2, [1, 0, 0]),
                (3, [1, 0, 0]),
                (4, [1, 0, 0]),
            ],
            [
                (1, [0, 0, 1]),
                (4, [-1, 0, 0]),
                (3, [-1, 0, 0]),
                (2, [0, 0, 0]),
                (3, [0, 0, 0]),
                (4, [0, 0, 0]),
            ],
            [
                (1, [0, 0, 1]),
                (2, [0, -1, 1]),
                (3, [0, -1, 1]),
                (2, [1, -1, 1]),
                (1, [1, 0, 1]),
                (4, [0, 0, 0]),
            ],
            [
                (1, [0, 0, 0]),
                (2, [0, -1, 0]),
                (3, [0, -1, 0]),
                (4, [0, 0, 0]),
                (3, [0, 0, 0]),
                (2, [0, 0, 0]),
            ],
            [
                (1, [1, 0, 0]),
                (2, [1, -1, 0]),
                (3, [0, -1, 0]),
                (4, [0, 0, 0]),
                (3, [0, 0, 0]),
                (2, [1, 0, 0]),
            ],
            [
                (1, [0, -1, 1]),
                (2, [0, -1, 1]),
                (1, [0, 0, 1]),
                (4, [0, 0, 0]),
                (3, [0, -1, 0]),
                (4, [0, -1, 0]),
            ],
            [
                (1, [0, 0, 1]),
                (2, [0, 0, 1]),
                (1, [0, 1, 1]),
                (4, [0, 1, 0]),
                (3, [0, 0, 0]),
                (4, [0, 0, 0]),
            ],
            [
                (1, [0, 0, 1]),
                (2, [0, 0, 1]),
                (3, [0, 0, 1]),
                (2, [1, 0, 1]),
                (1, [1, 0, 1]),
                (4, [0, 0, 0]),
            ],
            [
                (1, [1, -1, 1]),
                (2, [1, -1, 1]),
                (1, [1, 0, 1]),
                (4, [0, 0, 0]),
                (3, [0, -1, 0]),
                (4, [0, -1, 0]),
            ],
            [
                (1, [1, 0, 1]),
                (2, [1, 0, 1]),
                (1, [1, 1, 1]),
                (4, [0, 1, 0]),
                (3, [0, 0, 0]),
                (4, [0, 0, 0]),
            ],
        ],
    ]

    return rcount, rings_per_node, rnodes


@pytest.mark.parametrize(
    "strong_flag,label_key",
    [
        (False, "rings"),
        (True, "strong_rings"),
    ],
)
@patch("topo_metrics._julia_wrappers.run_rings", side_effect=patched_run_rings)
def test_get_clusters_with_mocked_rs_run_rings(
    mock_run_rings, sample_topology, strong_flag, label_key
):
    """Test Topology.get_clusters with mocked run_rings."""

    topology = sample_topology
    topology.properties.clear()

    result = topology.get_clusters(depth=12, strong=strong_flag)

    assert isinstance(result, RingsResults)
    assert result.rings_are_strong is strong_flag
    assert result.depth == 12
    assert isinstance(result.ring_size_count, RingSizeCounts)

    # Expected: 8 rings of size 6
    expected_ring_counts = np.zeros(27, dtype=int)
    expected_ring_counts[5] = 8  # 6-membered rings at index 5

    np.testing.assert_array_equal(
        result.ring_size_count.counts, expected_ring_counts
    )

    # Check caching
    assert (label_key, 12) in topology.properties
    result_cached = topology.get_clusters(depth=12, strong=strong_flag)
    assert result_cached is result


@patch(
    "topo_metrics._julia_wrappers.get_topological_genome", return_value="dia"
)
def test_get_topological_genome_mocked(mock_get_genome, sample_topology):
    """Test get_topological_genome with the Julia function mocked out."""

    topology: Topology = sample_topology
    topology.properties.pop("topological_genome", None)  # 💥 Force re-eval

    result = topology.get_topological_genome()

    assert result == "dia"
    assert topology.properties["topological_genome"] == "dia"

    result2 = topology.get_topological_genome()  # 💥 Cache
    assert result2 == result


def test_topology_repr(sample_topology):
    """Test the __repr__ method of Topology."""

    repr_str = repr(sample_topology)
    assert "Topology" in repr_str
    assert "nodes" in repr_str
    assert "edges" in repr_str
    assert "has_lattice" in repr_str


def test_rings_results_repr_single_vertex_symbol():
    """Test RingsResults __repr__ with a single vertex symbol."""
    from topo_metrics.clusters import Cluster
    from topo_metrics.rings import Ring, RingSizeCounts
    from topo_metrics.topology import RingsResults

    # Create a simple cluster with one ring
    nodes = [
        Node(node_id=i, frac_coord=np.random.rand(3)) for i in range(1, 4)
    ]
    ring = Ring(nodes=nodes, angle=((1, (0, 0, 0)), (2, (0, 0, 0))))
    cluster = Cluster(central_node_id=1, rings=[ring])

    result = RingsResults(
        depth=10,
        rings_are_strong=False,
        ring_size_count=RingSizeCounts(
            sizes=np.array([3]), counts=np.array([1])
        ),
        clusters=[cluster],
    )

    repr_str = repr(result)
    assert "RingsResults" in repr_str
    assert "depth" in repr_str
    assert "strong_rings" in repr_str
    assert "ring_size_count" in repr_str
    assert "VertexSymbol" in repr_str


def test_rings_results_repr_multiple_vertex_symbols():
    """Test RingsResults __repr__ with multiple different vertex symbols."""
    from topo_metrics.clusters import Cluster
    from topo_metrics.rings import Ring, RingSizeCounts
    from topo_metrics.topology import RingsResults

    # Create clusters that will have different vertex symbols
    nodes1 = [
        Node(node_id=i, frac_coord=np.random.rand(3)) for i in range(1, 4)
    ]
    nodes2 = [
        Node(node_id=i, frac_coord=np.random.rand(3)) for i in range(1, 5)
    ]

    ring1 = Ring(nodes=nodes1, angle=((1, (0, 0, 0)), (2, (0, 0, 0))))
    ring2 = Ring(nodes=nodes2, angle=((1, (0, 0, 0)), (2, (0, 0, 0))))

    cluster1 = Cluster(central_node_id=1, rings=[ring1])
    cluster2 = Cluster(central_node_id=2, rings=[ring2])

    result = RingsResults(
        depth=10,
        rings_are_strong=False,
        ring_size_count=RingSizeCounts(
            sizes=np.array([3, 4]), counts=np.array([1, 1])
        ),
        clusters=[cluster1, cluster2],
    )

    repr_str = repr(result)
    assert "RingsResults" in repr_str
    assert "VertexSymbols" in repr_str  # Plural because multiple


def test_rings_results_repr_many_vertex_symbols():
    """Test RingsResults __repr__ with many vertex symbols (truncate)."""
    from topo_metrics.clusters import Cluster
    from topo_metrics.rings import Ring, RingSizeCounts
    from topo_metrics.topology import RingsResults

    # Create 15 different clusters to test truncation at >10
    clusters = []
    for i in range(1, 16):
        nodes = [
            Node(node_id=j, frac_coord=np.random.rand(3))
            for j in range(1, i + 3)
        ]
        ring = Ring(nodes=nodes, angle=((1, (0, 0, 0)), (2, (0, 0, 0))))
        clusters.append(Cluster(central_node_id=i, rings=[ring]))

    result = RingsResults(
        depth=10,
        rings_are_strong=False,
        ring_size_count=RingSizeCounts(
            sizes=np.arange(3, 18), counts=np.ones(15, dtype=int)
        ),
        clusters=clusters,
    )

    repr_str = repr(result)
    assert "RingsResults" in repr_str
    assert "..." in repr_str  # Should show ellipsis for truncation


def test_topology_from_ase():
    """Test creating Topology from ASE Atoms object."""
    import ase

    # Create a simple cubic structure
    atoms = ase.Atoms(
        symbols=["Si", "Si"],
        positions=[[0, 0, 0], [1.5, 0, 0]],
        cell=[3, 3, 3],
        pbc=True
    )

    topology = Topology.from_ase(atoms, cutoff=2.0)

    assert len(topology.nodes) == 2
    assert topology.lattice is not None
    assert len(topology.edges) > 0


def test_topology_from_ase_with_removal():
    """Test Topology.from_ase with atom removal."""
    import ase

    # Create structure with O atoms to remove
    atoms = ase.Atoms(
        symbols=["Si", "O", "Si"],
        positions=[[0, 0, 0], [1.5, 0, 0], [3.0, 0, 0]],
        cell=[6, 6, 6],
        pbc=True
    )

    topology = Topology.from_ase(atoms, cutoff=2.0, remove_types={"O"})

    # O should be removed
    assert len(topology.nodes) == 2
    assert all(node.node_type == "Si" for node in topology.nodes)


def test_topology_from_ase_no_pbc():
    """Test Topology.from_ase with non-periodic structure."""
    import ase

    atoms = ase.Atoms(
        symbols=["C", "C"],
        positions=[[0, 0, 0], [1.5, 0, 0]],
        pbc=False  # No periodic boundary conditions
    )

    topology = Topology.from_ase(atoms, cutoff=2.0)

    assert topology.lattice is None  # No lattice for non-periodic


def test_node_post_init():
    """Test Node __post_init__ sets defaults correctly."""

    # Test with only node_id
    node = Node(node_id=1)
    assert node.node_type == "Si"
    assert node.frac_coord is None
    assert node.cart_coord is None
    assert node.is_shifted is False

    # Test with all parameters
    node2 = Node(
        node_id=2,
        node_type="C",
        frac_coord=np.array([0.5, 0.5, 0.5]),
        cart_coord=np.array([1, 1, 1]),
        is_shifted=True
    )
    assert node2.node_type == "C"
    assert node2.is_shifted is True
