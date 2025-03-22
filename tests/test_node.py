import numpy as np
import pytest
from pymatgen.core import Lattice as PymatgenLattice

from topo_metrics.topology import Node, get_all_node_frac_coords


@pytest.fixture
def sample_lattice():
    """Fixture to create a sample cubic lattice."""

    return PymatgenLattice.cubic(5.0)


def test_node_initialisation():
    """Test if Node initialises correctly with and without optional values."""

    node = Node(node_id=1)
    assert node.node_id == 1
    assert node.node_type == "Si"
    assert node.frac_coord is None
    assert node.cart_coord is None
    assert not node.is_shifted

    frac_coords = np.array([0.2, 0.4, 0.6])
    cart_coords = np.array([1.0, 2.0, 3.0])

    node_with_coords = Node(
        node_id=2, frac_coord=frac_coords, cart_coord=cart_coords
    )
    assert np.allclose(node_with_coords.frac_coord, frac_coords)
    assert np.allclose(node_with_coords.cart_coord, cart_coords)


def test_apply_image_shift(sample_lattice: PymatgenLattice):
    """Test shifting a node using apply_image_shift()."""

    frac_coords = np.array([0.2, 0.4, 0.6])
    node = Node(node_id=1, frac_coord=frac_coords)

    image_shift = np.array([1, 0, -1])
    shifted_node = node.apply_image_shift(sample_lattice, image_shift)

    expected_frac = frac_coords + image_shift
    expected_cart = sample_lattice.get_cartesian_coords(expected_frac)

    assert np.allclose(shifted_node.frac_coord, expected_frac)
    assert np.allclose(shifted_node.cart_coord, expected_cart)
    assert shifted_node.is_shifted


def test_apply_image_shift_no_coords(sample_lattice: PymatgenLattice):
    """Ensure apply_image_shift raises an error if no coordinates exist."""

    node = Node(node_id=1)
    image_shift = np.array([1, 0, -1])

    with pytest.raises(
        ValueError, match="Both `frac_coord` and `cart_coord` are missing"
    ):
        node.apply_image_shift(sample_lattice, image_shift)


def test_node_ordering():
    """Test that nodes are ordered by node_id."""

    node1 = Node(node_id=1)
    node2 = Node(node_id=2)
    node3 = Node(node_id=3)

    node_list = [node3, node1, node2]
    sorted_nodes = sorted(node_list)

    assert sorted_nodes == [node1, node2, node3]


def test_node_repr():
    """Test __repr__ for expected output."""

    node = Node(node_id=1, frac_coord=[0.1, 0.2, 0.3])
    repr_output = repr(node)

    assert "Node" in repr_output
    assert "node_id" in repr_output
    assert "frac_coord" in repr_output
    assert "cart_coord" not in repr_output


def test_apply_image_shift_with_cart_coords(sample_lattice: PymatgenLattice):
    """Test apply_image_shift() when only cartesian coordinates are provided."""

    cart_coords = np.array([1.0, 2.0, 3.0])
    node = Node(node_id=1, cart_coord=cart_coords)

    image_shift = np.array([1, -1, 0])
    shifted_node = node.apply_image_shift(sample_lattice, image_shift)

    # Confirm that frac_coord was computed
    expected_frac = (
        sample_lattice.get_fractional_coords(cart_coords) + image_shift
    )
    expected_cart = sample_lattice.get_cartesian_coords(expected_frac)

    assert np.allclose(shifted_node.frac_coord, expected_frac)
    assert np.allclose(shifted_node.cart_coord, expected_cart)
    assert shifted_node.is_shifted


def test_get_all_node_frac_coords():
    """Test that fractional coordinates are correctly extracted from nodes."""

    nodes = [
        Node(node_id=1, frac_coord=[0.1, 0.2, 0.3]),
        Node(node_id=2, frac_coord=[0.4, 0.5, 0.6]),
        Node(node_id=3, frac_coord=[0.7, 0.8, 0.9]),
    ]

    expected = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ],
        dtype=np.float64,
    )

    result = get_all_node_frac_coords(nodes)

    np.testing.assert_allclose(result, expected)
