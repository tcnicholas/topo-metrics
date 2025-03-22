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
def test_get_rings_with_mocked_rs_run_rings(
    mock_run_rings, sample_topology, strong_flag, label_key
):
    """Test Topology.get_rings with mocked run_rings."""

    topology = sample_topology
    topology.properties.clear()

    result = topology.get_rings(depth=12, strong=strong_flag)

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
    result_cached = topology.get_rings(depth=12, strong=strong_flag)
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
