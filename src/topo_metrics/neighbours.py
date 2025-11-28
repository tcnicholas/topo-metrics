from __future__ import annotations

from collections import deque
from typing import Any, Iterable

import numpy as np
import numpy.typing as npt


def autoreduce_neighborlist(
    frac_coords: npt.NDArray[np.float64],
    symbols: list[str],
    edges: npt.NDArray[np.int_],
    remove_types: Iterable[Any] | None = None,
    remove_degree2: bool = False
) -> tuple[
    npt.NDArray[np.float64],
    list[str],
    npt.NDArray[np.int_],
    npt.NDArray[np.int_]
]:
    """
    Simplify a periodic bonded graph by contracting out selected atoms.

    Parameters
    ----------
    frac_coords
        Fractional coordinates of all atoms.
    symbols
        Atomic symbols (length N).
    edges
        Columns: i+1, j+1, Sx, Sy, Sz (1-based atom indices).
    remove_types
        If not None, atoms whose symbol is in this set are removed and their
        neighbors are connected together (clique over neighbors).
        Example: {"O"}.
    remove_degree2
        If True, atoms that are 2-connected are also removed (in addition to
        any atoms in `remove_types`).

    Returns
    -------
    new_frac_coords : (N_keep,3) ndarray
    new_symbols : list[str] length N_keep
    new_edges : (M_new,5) int ndarray
        Same format as input `edges` (1-based indices).
    old_to_new : (N,) ndarray of int
        Mapping from old atom index (0-based) to new index (0-based). -1 for
        removed atoms.
    """

    frac_coords = np.asarray(frac_coords)
    edges = np.asarray(edges, dtype=int)
    N = frac_coords.shape[0]

    # 0-based indices for manipulation
    i0 = edges[:, 0] - 1
    j0 = edges[:, 1] - 1
    S = edges[:, 2:5].astype(int)

    # Build adjacency: adjacency[u] = list of (v, S_uv) where S_uv is shift u->v
    adjacency = [[] for _ in range(N)]
    for u, v, s in zip(i0, j0, S):
        s_vec = np.asarray(s, dtype=int)
        adjacency[u].append((v, s_vec))
        adjacency[v].append((u, -s_vec))

    symbols = list(symbols)
    remove_types_set = set(remove_types) if remove_types is not None else set()
    removable_by_type = np.array([
        sym in remove_types_set for sym in symbols],
        dtype=bool
    )

    degrees = np.array([len(adj) for adj in adjacency], dtype=int)
    removed = np.zeros(N, dtype=bool)
    scheduled = np.zeros(N, dtype=bool)

    # Initial queue: atoms selected by type
    q = deque()
    for idx in range(N):
        if removable_by_type[idx] and degrees[idx] > 0:
            q.append(idx)
            scheduled[idx] = True

    # Also: initial degree-2 atoms, if requested
    if remove_degree2:
        for idx in range(N):
            if not scheduled[idx] and degrees[idx] == 2:
                q.append(idx)
                scheduled[idx] = True

    def add_edge(u, v, S_uv) -> None:
        """Add undirected edge u<->v with shift S_uv from u to v."""
    
        adjacency[u].append((v, S_uv))
        adjacency[v].append((u, -S_uv))
        degrees[u] += 1
        degrees[v] += 1

    # Main contraction loop
    while q:
        r = q.popleft()
        if removed[r]:
            continue

        # Current neighbors that are still alive
        neighbors = [(n, s.copy()) for (n, s) in adjacency[r] if not removed[n]]

        # Connect neighbors pairwise through r
        k = len(neighbors)
        if k >= 2:
            for ia in range(k):
                a, S_ra = neighbors[ia]
                for ib in range(ia + 1, k):
                    b, S_rb = neighbors[ib]
                    if a == b:
                        continue
                    # S_ab (a -> b) from r->a and r->b is: S_ab = -S_ra + S_rb
                    S_ab = -S_ra + S_rb
                    add_edge(a, b, S_ab)

        # Remove r from neighbors' adjacency lists
        for n, _ in neighbors:
            old_list = adjacency[n]
            if not old_list:
                continue
            new_list = [(nbr, s) for (nbr, s) in old_list if nbr != r]
            removed_count = len(old_list) - len(new_list)
            if removed_count:
                adjacency[n] = new_list
                degrees[n] -= removed_count

                # Newly 2-connected atoms can be scheduled if we are doing
                # degree-2 reduction and they are not type-protected
                if (remove_degree2 and not removed[n] and
                        not removable_by_type[n] and
                        degrees[n] == 2 and not scheduled[n]):
                    q.append(n)
                    scheduled[n] = True

        # Finally mark r as removed
        adjacency[r] = []
        degrees[r] = 0
        removed[r] = True

    # Build mapping old index -> new index for surviving atoms
    old_to_new = -np.ones(N, dtype=int)
    keep_indices = [i for i in range(N) if not removed[i]]
    for new_idx, old_idx in enumerate(keep_indices):
        old_to_new[old_idx] = new_idx

    # Rebuild edge list from adjacency of surviving atoms (deduplicated)
    edge_keys = set()
    for u in keep_indices:
        for v, S_uv in adjacency[u]:
            if removed[v]:
                continue

            # Canonical orientation: lower index first, adjust shift sign
            if u <= v:
                key = (u, v, int(S_uv[0]), int(S_uv[1]), int(S_uv[2]))
            else:
                key = (v, u, int(-S_uv[0]), int(-S_uv[1]), int(-S_uv[2]))
            edge_keys.add(key)

    if edge_keys:
        edge_array = np.array(sorted(edge_keys), dtype=int)
        i_new = old_to_new[edge_array[:, 0]]
        j_new = old_to_new[edge_array[:, 1]]
        S_new = edge_array[:, 2:5]
        new_edges = np.column_stack((i_new + 1, j_new + 1, S_new)).astype(int)
    else:
        new_edges = np.zeros((0, 5), dtype=int)

    new_frac_coords = frac_coords[keep_indices]
    new_symbols = [symbols[i] for i in keep_indices]

    return new_frac_coords, new_symbols, new_edges, old_to_new
