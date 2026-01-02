from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from pymatgen.core.lattice import Lattice


@dataclass
class Configuration:
    """A configuration parsed from a conflink file."""

    positions: npt.NDArray[np.float64]
    box_size: float
    connections: dict

    def to_nodes_edges_lattice(self, node_type: str = "Si") -> dict:
        """Convert this configuration to a Topology object."""

        lattice = Lattice.cubic(self.box_size)
        frac_coords = lattice.get_fractional_coords(self.positions)
        cart_coords = lattice.get_cartesian_coords(frac_coords)

        edges = []
        for atom_id, conn_ids in self.connections.items():
            conn_ids = sorted(conn_ids)
            frac1 = frac_coords[atom_id]
            for conn_id in conn_ids:
                frac2 = frac_coords[conn_id]
                _, im = lattice.get_distance_and_image(frac1, frac2)
                edges.append(
                    np.array((atom_id + 1, conn_id + 1, *im), dtype=int)
                )

        edges = np.array(edges, dtype=int)

        return {
            "cart_coords": cart_coords,
            "frac_coords": frac_coords,
            "edges": edges,
            "lattice": lattice,
        }


def parse_conflink(filename: str) -> list[Configuration]:
    """Parse one or more configurations from a conflink file."""

    with open(filename) as file:
        lines = [ln.strip() for ln in file if ln.strip()]

    configs = []
    i = 0
    n_lines = len(lines)

    while i < n_lines:
        header_parts = [p.strip() for p in lines[i].split(",") if p.strip()]
        n_particles = int(header_parts[0])
        box_size = float(header_parts[1])
        i += 1

        positions = np.empty((n_particles, 3), dtype=float)
        for p_idx in range(n_particles):
            parts = [p.strip() for p in lines[i].split(",") if p.strip()]
            positions[p_idx] = [float(x) for x in parts]
            i += 1

        connections = defaultdict(set)
        for _ in range(n_particles):
            header_parts = [p.strip() for p in lines[i].split(",") if p.strip()]
            atom_id = int(header_parts[0])
            i += 1

            conn_parts = [p.strip() for p in lines[i].split(",") if p.strip()]

            for conn_id_str in conn_parts:
                conn_id = int(conn_id_str)
                connections[atom_id].add(conn_id)
                connections[conn_id].add(atom_id)

            i += 1

        configs.append(Configuration(positions, box_size, connections))

    return configs
