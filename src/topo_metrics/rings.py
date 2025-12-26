from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, NamedTuple

import numpy as np
import numpy.typing as npt

from topo_metrics.ring_geometry import RingGeometry
from topo_metrics.utils import to_tuple, uniform_repr

if TYPE_CHECKING:  # pragma: no cover
    from topo_metrics.topology import Node, Topology


@dataclass(frozen=True)
class RingSizeCounts:
    sizes: npt.NDArray[np.int_]
    counts: npt.NDArray[np.int_]

    def __getitem__(self, key: int) -> tuple[int, int]:
        """Get the ring size and count at the specified index."""

        return self.sizes[key], self.counts[key]

    def __iter__(self) -> Iterator[tuple[int, int]]:
        """Iterate through the ring sizes and counts."""

        return ((self.sizes[i], self.counts[i]) for i in range(len(self.sizes)))

    def __repr__(self) -> str:
        """Return a string representation of the RingSizeCounts object."""

        nonzero_idx = np.where(self.counts > 0)[0]

        if len(nonzero_idx) == 0:
            # Edge case: no rings or all counts are zero.
            info = {
                "n_rings": 0,
                "min_ring_size": None,
                "max_ring_size": None,
            }
        else:
            # Minimum and maximum sizes where counts are non-zero
            min_ring_size = int(self.sizes[nonzero_idx].min())
            max_ring_size = int(self.sizes[nonzero_idx].max())

            info = {
                "n_rings": int(self.counts.sum()),
                "min": min_ring_size,
                "max": max_ring_size,
            }

        return uniform_repr(
            "RingSizeCounts", **info, stringify=True, indent_size=4
        )


class Ring(NamedTuple):
    nodes: list[Node]
    """ The node IDs that form the ring. Neighbouring nodes are connected by an 
    edge, and the last node is connected to the first node. """

    angle: tuple[tuple[int, tuple[int]]]
    """ 
    Neighbouring nodes about the central node, listed with node ID and image. 
    """

    @property
    def size(self) -> int:
        """Ring size."""
        return len(self)

    def geometry(self) -> RingGeometry:
        """Get the ring geometry object."""
        return RingGeometry(nodes=tuple(self.nodes))

    def __len__(self) -> int:
        """Ring size."""
        return len(self.nodes)

    def __repr__(self) -> str:
        info = {"n": len(self.nodes)}
        return uniform_repr("Ring", **info, stringify=True, indent_size=4)


############################### CONVERSIONS ###############################


def get_ordered_node_list(
    nodes_and_images: list[tuple[int, npt.NDArray[np.int_]]],
    central_node_id: int,
) -> list[tuple[int, npt.NDArray[np.int_]]]:
    """
    Sorts a cyclic list of nodes and images, placing the central_node_id first
    and preserving cyclic order while ensuring a deterministic ordering.

    Parameters
    ----------
    nodes_and_images
        A list of node IDs and image shifts.
    central_node_id
        The node ID of the central node.

    Returns
    -------
    A sorted list of node IDs and image shifts.
    """

    def should_reverse(ring: list[tuple[int, npt.NDArray[np.int_]]]) -> bool:
        """
        Determines if the ring should be reversed by checking node ID order.
        """

        before = ring[-1]
        after = ring[1]

        if before[0] < after[0]:  # If previous ID is smaller, reverse
            return True
        elif before[0] > after[0]:  # If next ID is smaller, keep order
            return False

        # If node IDs are equal, compare their image values
        before = ring[-1]
        after = ring[1]

        return tuple(before[1]) < tuple(after[1])

    central_index: int | None = None
    for i, (node, image) in enumerate(nodes_and_images):
        if node == central_node_id and np.array_equal(image, [0, 0, 0]):
            central_index = i
            break

    if central_index is None:
        raise ValueError("Central node with image [0, 0, 0] not found")

    # Rotate so the central node is first
    sorted_ring = (
        nodes_and_images[central_index:] + nodes_and_images[:central_index]
    )

    # Check if we need to reverse order to make it deterministic
    if should_reverse(sorted_ring):
        sorted_ring.reverse()
        sorted_ring = [sorted_ring[-1]] + sorted_ring[:-1]

    return sorted_ring


def node_list_to_ring(
    topology: Topology,
    node_list: list[tuple[int, npt.NDArray[np.int_]]],
    central_node_id: int,
) -> Ring:
    """Convert a list of node IDs and image shifts to a Ring object.

    Parameters
    ----------
    topology
        The topology object containing the nodes.
    node_list
        A list of node IDs and image shifts.
    central_node_id
        The node ID of the central node.

    Returns
    -------
    A Ring object with the nodes in the correct positions.
    """

    assert topology.lattice is not None

    # the first entry is the central node.
    ordered_node_list = get_ordered_node_list(node_list, central_node_id)

    # get the nodes in the correct positions.
    shifted_nodes = [
        topology[node_id].apply_image_shift(topology.lattice, shift)
        for node_id, shift in ordered_node_list
    ]

    # the angle is defined for this central node by the neighbouring two nodes.
    angle = to_tuple([ordered_node_list[-1], ordered_node_list[1]])

    return Ring(nodes=shifted_nodes, angle=angle)
