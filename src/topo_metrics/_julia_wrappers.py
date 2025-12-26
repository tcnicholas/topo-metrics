from __future__ import annotations

import numpy as np
import numpy.typing as npt

from topo_metrics.paths import RingStatistics as RS


def run_rings(edges, depth):
    rcount, _, rnodes = RS.run_rings(edges, depth)
    rcount = np.array(rcount)
    converted_rnodes: list[list[list[tuple[int, npt.NDArray[np.int_]]]]] = [
        [
            [(int(t[0]), np.asarray(t[1], dtype=np.int_)) for t in ring]
            for ring in node
        ]
        for node in rnodes
    ]
    return rcount, converted_rnodes


def run_strong_rings(edges, depth):
    rcount, _, rnodes = RS.run_strong_rings(edges, depth)
    rcount = np.array(rcount)
    converted_rnodes: list[list[list[tuple[int, npt.NDArray[np.int_]]]]] = [
        [
            [(int(t[0]), np.asarray(t[1], dtype=np.int_)) for t in ring]
            for ring in node
        ]
        for node in rnodes
    ]
    return rcount, converted_rnodes


def get_topological_genome(nodes, edges, cell_lengths, cell_angles):
    return RS.get_topological_genome(nodes.T, edges, cell_lengths, cell_angles)


def get_all_rings(edges, depth):
    all_rings = RS.get_all_rings(edges, depth)
    rings = [
        [(int(t[0]), np.asarray(t[1], dtype=np.int_)) for t in ring]
        for ring in all_rings
    ]
    return rings


def get_coordination_sequences(edges, dmax):
    all_coord_seqs = RS.run_coordination_sequences(edges, dmax)
    coord_seqs = [[int(t) for t in seq] for seq in all_coord_seqs]
    return np.array(coord_seqs)
