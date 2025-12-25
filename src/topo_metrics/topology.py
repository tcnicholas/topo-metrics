from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Hashable, Iterable, NamedTuple

import ase
import numpy as np
import numpy.typing as npt
from pymatgen.core.lattice import Lattice as PymatgenLattice

from topo_metrics._julia_wrappers import (
    get_all_rings,
    get_coordination_sequences,
    get_topological_genome,
    run_rings,
    run_strong_rings,
)
from topo_metrics.clusters import (
    Cluster,
    get_carvs_vector,
    get_clusters,
    get_vertex_symbol,
)
from topo_metrics.io.cgd import parse_cgd, process_neighbour_list
from topo_metrics.neighbours import (
    autoreduce_neighborlist,
    graph_edges_by_cutoff,
)
from topo_metrics.ring_geometry import RingGeometry
from topo_metrics.rings import RingSizeCounts
from topo_metrics.symbols import VertexSymbol
from topo_metrics.utils import uniform_repr


class RingsResults(NamedTuple):
    depth: int
    """ The depth to which the rings were searcher for. """

    rings_are_strong: bool
    """ Whether the rings were filtered to strong rings only. """

    ring_size_count: RingSizeCounts
    """
    The number of rings of a given size, of shape ``(**, 2)`` where the first
    column indicates the ring size, and the second indicates the number of rings
    of that size.
    """

    clusters: list[Cluster]
    """
    Each node can be characterised in terms of the rings in which it 
    participates. This can be summarised using common metrics such as Vertex 
    Symbols.
    """

    def __repr__(self) -> str:
        vs_name = "VertexSymbol"
        vertex_symbols_str = ""

        vertex_symbols = {
            x.to_str() for x in get_vertex_symbol(self.clusters)
            if isinstance(x, VertexSymbol)
        }

        if len(vertex_symbols) > 1:
            vs_name = "VertexSymbols"
            vertex_symbols_str += "{\n\t"

        if len(vertex_symbols) > 10:
            displayed_symbols = (
                list(vertex_symbols)[:5] + ["..."] + list(vertex_symbols)[-5:]
            )
        else:
            displayed_symbols = vertex_symbols

        vertex_symbols_str += ",\n\t".join(displayed_symbols)

        if len(vertex_symbols) > 1:
            vertex_symbols_str += "\n}"

        info = {
            "depth": self.depth,
            "strong_rings": self.rings_are_strong,
            "ring_size_count": self.ring_size_count,
            vs_name: vertex_symbols_str,
            "CARVS": get_carvs_vector(self.clusters),
        }

        return uniform_repr(
            "RingsResults", **info, indent_size=4, stringify=False
        )


@dataclass
class Topology:
    """
    A class detailing the topology of a network, based on nodes and edges.
    """

    nodes: list[Node]
    edges: npt.NDArray[np.int_] = field(
        default_factory=lambda: np.empty((0, 5), dtype=int)
    )
    lattice: PymatgenLattice | None = None
    properties: dict[Hashable, Any] = field(default_factory=dict)

    @classmethod
    def from_ase(
        cls,
        ase_atoms: ase.Atoms, 
        cutoff: float = 0.0,
        pair_cutoffs: dict[tuple[str, str], float] | None = None,
        remove_types: Iterable[Any] | None = None,
        remove_degree2: bool = False
    ) -> Topology:
        """
        Creates a Topology object from an ASE Atoms object.

        Parameters
        ----------
        ase_atoms
            The ASE Atoms object representing the structure.

        Returns
        -------
        A Topology object representing the network as nodes and edges.
        """

        lattice = None
        if all(ase_atoms.pbc):
            lattice = PymatgenLattice(ase_atoms.cell)

        frac_coords = ase_atoms.get_scaled_positions()
        symbols = ase_atoms.get_chemical_symbols()

        edges = graph_edges_by_cutoff(
            ase_atoms,
            cutoff=cutoff,
            pair_cutoffs=pair_cutoffs,
            one_based=True
        )

        if remove_types is not None or remove_degree2:
            frac_coords, symbols, edges, _ = autoreduce_neighborlist(
                frac_coords,
                symbols,
                edges,
                remove_types=remove_types,
                remove_degree2=remove_degree2
            )
        
        nodes = []
        for idx, (frac,symbol) in enumerate(zip(frac_coords, symbols), start=1):
            nodes.append(Node(node_id=idx, node_type=symbol, frac_coord=frac))

        return cls(nodes=nodes, edges=edges, lattice=lattice)

    @classmethod
    def from_cgd(cls, filename: str) -> Topology:
        """
        Parses and loads a CGD file with an adjacency matrix.

        Parameters
        ----------
        filename
            The path to the CGD file.

        Returns
        -------
        A Topology object representing the network as nodes and edges.
        """

        if not os.path.exists(filename):
            raise FileNotFoundError(f"File '{filename}' not found.")

        lattice, atom_labels, atoms, edges = parse_cgd(filename)
        neighbour_list = process_neighbour_list(edges, atoms, atom_labels)

        # Create Node instances
        all_nodes = [
            Node(
                node_id=i + 1,
                node_type=label,
                frac_coord=frac_coord if atoms is not None else None,
            )
            for i, (label, frac_coord) in enumerate(zip(atom_labels, atoms))
            if atoms is not None
        ]

        return cls(nodes=all_nodes, edges=neighbour_list, lattice=lattice)
    
    def get_rings(
        self,
        depth: int = 12
    ) -> list[RingGeometry]:
        """Computes or retrieves unique rings in the network.

        Parameters
        ----------
        depth
            The maximum depth to search for rings.

        Notes
        -----
        - In the previous implementation, this method returned the clusters of 
          rings at each node. This is obtained instead via the `get_clusters`
          method. This method now returns all unique rings in the network as
          RingGeometry objects.
        """

        if self.lattice is None:
            raise ValueError(
                "Currently, ring geometries require a defined lattice. This "
                "will be supported in future releases because it is not a "
                "fundamental requirement for ring finding."
            )

        rings = get_all_rings(self.edges, depth)

        final_rings = []
        for ring in rings:
            this_ring = []
            for node_id, image in ring:
                node = self.nodes[node_id-1]
                node = node.apply_image_shift(self.lattice, image)
                this_ring.append(node)
            final_rings.append(RingGeometry(tuple(this_ring)))
        
        return final_rings

    def get_clusters(
        self,
        depth: int = 12,
        strong: bool = False
    ) -> RingsResults:
        """Computes or retrieves ring statistics for the network.

        Parameters
        ----------
        depth
            The maximum depth to search for rings.
        strong
            Whether to filter the rings to strong rings only.

        Returns
        -------
        A dictionary containing the ring statistics.
        """

        label = ("strong_rings" if strong else "rings", depth)

        # check if cached results exist and return them.
        if label in self.properties:
            result = self.properties[label]
            assert isinstance(result, RingsResults)
            return result

        # compute rings.
        compute_rings = run_strong_rings if strong else run_rings
        rcount, rnodes = compute_rings(self.edges, depth)

        # store and return results.
        results = RingsResults(
            depth=depth,
            rings_are_strong=strong,
            ring_size_count=RingSizeCounts(*rcount.T),
            clusters=get_clusters(self, rnodes),
        )

        self.properties[label] = results

        return results

    def get_topological_genome(self) -> str:
        """
        Returns a the topology code for the framework.

        Notes
        -----
        - The topological genome is a finite series of numbers that is provably
          unique for each net.
        - It can be comptued in polynomial time with respect to the size of the
          net.
        """

        if "topological_genome" in self.properties:
            return str(self.properties["topological_genome"])

        nodes = get_all_node_frac_coords(self.nodes)
        assert self.lattice is not None, (
            "Lattice must be defined to compute genome."
        )
        cell_lengths = self.lattice.lengths
        cell_angles = self.lattice.angles

        topology_genome = get_topological_genome(
            nodes,
            self.edges,
            cell_lengths,
            cell_angles,
        )

        self.properties["topological_genome"] = topology_genome

        return topology_genome
    
    def get_coordination_sequences(
        self,
        max_shell: int = 10,
        node_ids: Iterable[int] | int | None = None
    ) -> npt.NDArray[np.int_]:
        """
        Return coordination sequences for specified nodes.

        Parameters
        ----------
        max_shell
            The maximum shell to compute coordination sequences to.
        node_ids
            The node IDs for which to compute coordination sequences. If None,
            coordination sequences for all nodes are returned.
        """

        cs = get_coordination_sequences(self.edges, max_shell)

        if node_ids is None:
            return cs

        if isinstance(node_ids, int):
            node_ids = [node_ids]

        return cs[np.array(node_ids, dtype=int) - 1]

    def __getitem__(self, node_id: int) -> Node:
        """Retrieve a Node object by its node number."""
        if not (1 <= node_id <= len(self.nodes)):
            raise IndexError(
                f"node_id {node_id} is out of bounds "
                f"(valid range: 1 to {len(self.nodes)})"
            )

        return self.nodes[node_id - 1]

    def __repr__(self) -> str:
        info = {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "has_lattice": self.lattice is not None,
        }
        return uniform_repr("Topology", **info, indent_size=4)


@dataclass(order=True)
class Node:
    """
    A representation of a node in a network.
    """

    node_id: int
    node_type: str | None = "Si"
    frac_coord: npt.NDArray[np.floating] | None = field(default=None)
    cart_coord: npt.NDArray[np.floating] | None = field(default=None)
    is_shifted: bool = field(default=False)

    def apply_image_shift(
        self,
        lattice: PymatgenLattice,
        image_shift: npt.NDArray[np.int_],
    ) -> Node:
        """Apply the image shift to this node and return a new Node object.

        Parameters
        ----------
        lattice
            The lattice object for the network.
        image_shift
            The shift vector to apply to the node coordinates.

        Returns
        -------
        A new Node object with the shifted coordinates.
        """

        if self.frac_coord is None and self.cart_coord is None:
            raise ValueError(
                "Both `frac_coord` and `cart_coord` are missing; "
                "cannot compute shifted coordinates."
            )

        if self.frac_coord is not None:
            frac_coord = self.frac_coord
        else:
            assert self.cart_coord is not None
            frac_coord = lattice.get_fractional_coords(self.cart_coord)

        shifted_frac = (frac_coord + image_shift).astype(np.float64)
        shifted_cart = lattice.get_cartesian_coords(shifted_frac)

        return Node(
            node_id=self.node_id,
            node_type=self.node_type,
            frac_coord=shifted_frac,
            cart_coord=shifted_cart,
            is_shifted=True,
        )

    def __post_init__(self) -> None:
        """Ensure coordinates are NumPy arrays if provided."""

        self.sort_index = self.node_id
        if self.frac_coord is not None:
            self.frac_coord = np.array(self.frac_coord, dtype=np.float64)
        if self.cart_coord is not None:
            self.cart_coord = np.array(self.cart_coord, dtype=np.float64)

    def __repr__(self) -> str:
        name = "Node" if not self.is_shifted else "ShiftedNode"
        info = {"node_id": self.node_id, "node_type": self.node_type}

        if self.frac_coord is not None:
            formatted_coords = np.array2string(
                np.round(self.frac_coord, 2),
                precision=2,
                separator=", ",
                floatmode="fixed",
            )
            info["frac_coord"] = formatted_coords

        if self.cart_coord is not None:
            formatted_coords = np.array2string(
                np.round(self.cart_coord, 2),
                precision=2,
                separator=", ",
                floatmode="fixed",
            )
            info["cart_coord"] = formatted_coords

        return uniform_repr(name, **info, indent_size=4)


############################### HELPERS ###############################


def get_all_node_frac_coords(nodes: list[Node]) -> npt.NDArray[np.floating]:
    """Return the fractional coordinates of all nodes in the network."""

    return np.array([node.frac_coord for node in nodes], dtype=np.float64)
