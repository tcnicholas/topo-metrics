import numpy as np
import pytest

from topo_metrics.rings import (
    RingSizeCounts,
    get_ordered_node_list,
    node_list_to_ring,
)


def test_get_ordered_node_list_ordering():
    """Test that nodes are ordered by node_id."""

    ring = [
        (2, np.array([0, 0, 0])),
        (3, np.array([0, 0, 0])),
        (1, np.array([0, 0, 0])),
    ]
    ordered = get_ordered_node_list(ring, 1)

    assert ordered[0][0] == 1
    assert len(ordered) == 3


def test_get_ordered_node_list_missing_central():
    """Test that an error is raised when the central node is missing."""

    ring = [
        (2, np.array([0, 0, 0])),
        (3, np.array([0, 0, 0])),
    ]

    with pytest.raises(ValueError, match="Central node with image"):
        get_ordered_node_list(ring, 1)


def test_node_list_to_ring(sample_topology):
    """Test that a Ring object is created from a list of nodes."""

    node_list = [
        (1, np.array([0, 0, 0])),
        (2, np.array([0, 0, 0])),
        (3, np.array([0, 0, 0])),
    ]
    ring = node_list_to_ring(sample_topology, node_list, central_node_id=1)

    assert len(ring.nodes) == 3
    assert isinstance(ring.angle, tuple)
    assert hasattr(ring, "size")


def test_ring_size_counts_getitem():
    """Test that the RingSizeCounts object can be indexed."""

    sizes = np.array([4, 5, 6])
    counts = np.array([10, 20, 30])
    rsc = RingSizeCounts(sizes, counts)

    assert rsc[0] == (4, 10)
    assert rsc[1] == (5, 20)
    assert rsc[2] == (6, 30)


def test_ring_size_counts_iter():
    """Test that the RingSizeCounts object can be iterated."""

    sizes = np.array([3, 4])
    counts = np.array([1, 2])
    rsc = RingSizeCounts(sizes, counts)

    output = list(rsc)
    assert output == [(3, 1), (4, 2)]


########## TEST ORDERING ##########


def make_node(node_id, image):
    return (node_id, np.array(image, dtype=int))


def assert_node_lists_equal(result, expected):
    assert len(result) == len(expected)
    for (rid, rimg), (eid, eimg) in zip(result, expected):
        assert rid == eid
        assert np.array_equal(rimg, eimg)


def test_reverses_due_to_node_id_order():
    nodes = [
        make_node(2, [0, 0, 0]),
        make_node(4, [0, 0, 0]),
        make_node(1, [0, 0, 0]),  # central
        make_node(5, [0, 0, 0]),
    ]

    result = get_ordered_node_list(nodes, central_node_id=1)

    expected = [
        make_node(1, [0, 0, 0]),
        make_node(4, [0, 0, 0]),
        make_node(2, [0, 0, 0]),
        make_node(5, [0, 0, 0]),
    ]

    assert_node_lists_equal(result, expected)


def test_does_not_reverse_if_node_id_higher():
    nodes = [
        make_node(3, [0, 0, 0]),
        make_node(2, [0, 0, 0]),
        make_node(1, [0, 0, 0]),  # central
    ]

    result = get_ordered_node_list(nodes, central_node_id=1)
    expected = [
        make_node(1, [0, 0, 0]),
        make_node(2, [0, 0, 0]),
        make_node(3, [0, 0, 0]),
    ]

    assert_node_lists_equal(result, expected)


def test_reverse_due_to_image_values():
    nodes = [
        make_node(2, [0, 0, 1]),  # before: same ID, larger image shift
        make_node(2, [0, 0, 0]),  # after
        make_node(1, [0, 0, 0]),  # central
    ]

    result = get_ordered_node_list(nodes, central_node_id=1)

    expected = [
        make_node(1, [0, 0, 0]),
        make_node(2, [0, 0, 0]),
        make_node(2, [0, 0, 1]),
    ]

    assert_node_lists_equal(result, expected)


def test_ring_size_counts_repr_empty():
    """Test RingSizeCounts __repr__ with no rings."""

    sizes = np.array([4, 5, 6])
    counts = np.array([0, 0, 0])  # All zero counts
    rsc = RingSizeCounts(sizes, counts)

    repr_str = repr(rsc)
    assert "RingSizeCounts" in repr_str
    assert "n_rings" in repr_str
    assert "0" in repr_str  # Should show 0 rings


def test_ring_size_counts_repr_nonzero():
    """Test RingSizeCounts __repr__ with rings."""

    sizes = np.array([4, 5, 6])
    counts = np.array([10, 0, 20])
    rsc = RingSizeCounts(sizes, counts)

    repr_str = repr(rsc)
    assert "RingSizeCounts" in repr_str
    assert "n_rings" in repr_str
    assert "30" in repr_str  # Total rings
    assert "min" in repr_str
    assert "max" in repr_str


def test_ring_repr(sample_topology):
    """Test Ring __repr__ method."""
    from topo_metrics.rings import Ring

    node_list = [
        (1, np.array([0, 0, 0])),
        (2, np.array([0, 0, 0])),
        (3, np.array([0, 0, 0])),
    ]
    ring = node_list_to_ring(sample_topology, node_list, central_node_id=1)

    repr_str = repr(ring)
    assert "Ring" in repr_str
    assert "n" in repr_str


def test_ring_size_property(sample_topology):
    """Test Ring size property."""

    node_list = [
        (1, np.array([0, 0, 0])),
        (2, np.array([0, 0, 0])),
        (3, np.array([0, 0, 0])),
        (4, np.array([0, 0, 0])),
    ]
    ring = node_list_to_ring(sample_topology, node_list, central_node_id=1)

    assert ring.size == 4
    assert len(ring) == 4
