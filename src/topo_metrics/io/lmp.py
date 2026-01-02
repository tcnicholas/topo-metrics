from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from ase import Atoms

RingLocalLink = tuple[int, int, int, int, int]
"""
Ring-local link:

    (ri, li, rj, lj, bond_type)

where ri, rj are ring indices (0-based), li, lj are local indices within the 
rings (0-based), and bond_type is the LAMMPS bond type integer.
"""

GlobalLink = tuple[int, int, int]
"""
Global link:

    (gi, gj, bond_type)

where gi, gj are global atom indices (0-based), and bond_type is the LAMMPS bond
type integer.
"""


@dataclass(frozen=True)
class PackedRings:
    atoms: Atoms
    ring_ranges: list[range]
    ring_species: list[str]


def cycle_edges(indices_0based: Sequence[int]) -> list[tuple[int, int]]:
    idx = list(indices_0based)
    if len(idx) < 3:
        raise ValueError(f"Ring must have >= 3 vertices, got {len(idx)}")
    return [(idx[i], idx[(i + 1) % len(idx)]) for i in range(len(idx))]


def _ring_symbols(ring) -> list[str]:
    """Extract list of element symbols from a ring object."""

    if hasattr(ring, "nodes"):
        return [str(n.node_type) for n in ring.nodes]

    if hasattr(ring, "species"):
        syms = re.findall(r"[A-Z][a-z]?", str(ring.species))
        if syms:
            return syms

    raise TypeError(
        "Ring object must provide .nodes with .node_type, or a parsable "
        ".species string"
    )


def set_types_by_ring_and_element(
    packed: PackedRings,
) -> dict[tuple[int, str], int]:
    """
    Assign LAMMPS atom types so that each (ring_index, element_symbol) is a
    unique type.

    Returns:
        mapping[(ring_idx, symbol)] = type_id  (type_id is 1-based)
    """
    atoms = packed.atoms
    n = len(atoms)
    types = np.empty(n, dtype=int)

    mapping: dict[tuple[int, str], int] = {}
    next_type = 1

    symbols = atoms.get_chemical_symbols()

    for r_idx, r_range in enumerate(packed.ring_ranges):
        for gi in range(r_range.start, r_range.stop):
            sym = symbols[gi]
            key = (r_idx, sym)
            if key not in mapping:
                mapping[key] = next_type
                next_type += 1
            types[gi] = mapping[key]

    atoms.set_array("type", types)
    return mapping


def _add_undirected_once(neigh, i: int, j: int, t: int):
    """Store each undirected bond once on the lower index atom."""
    a, b = (i, j) if i < j else (j, i)
    neigh[a].append((b, t))


def adjacency_to_ase_bonds_array(
    neigh: list[list[tuple[int, int]]],
) -> np.ndarray:
    out = []
    for lst in neigh:
        if not lst:
            out.append("_")
        else:
            out.append(",".join(f"{j}({t})" for j, t in lst))
    return np.array(out, dtype=object)


def pack_rings_to_ase(
    rings: Sequence,
    *,
    cell: np.ndarray | None = None,
    pbc: bool | tuple[bool, bool, bool] = False,
    images: np.ndarray | None = None,
    image_convention: str = "subtract",
) -> PackedRings:
    """Packs a sequence of RingGeometry into a single ASE Atoms object."""

    n_rings = len(rings)
    if images is None:
        images = np.zeros((n_rings, 3), dtype=int)
    images = np.asarray(images, dtype=int)
    if images.shape != (n_rings, 3):
        raise ValueError(
            f"images must have shape (n_rings,3), got {images.shape}"
        )

    if cell is not None:
        cell = np.asarray(cell, float)
        if cell.shape != (3, 3):
            raise ValueError(f"cell must be (3,3), got {cell.shape}")

    all_symbols: list[str] = []
    all_positions: list[np.ndarray] = []
    ring_ranges: list[range] = []
    ring_species: list[str] = []

    offset = 0
    for k, ring in enumerate(rings):
        pos = np.asarray(ring.positions, float)
        if cell is not None:
            shift = images[k] @ cell  # (3,)
            if image_convention == "subtract":
                pos = pos - shift
            elif image_convention == "add":
                pos = pos + shift
            else:
                raise ValueError("image_convention must be 'subtract' or 'add'")

        symbols = _ring_symbols(ring)
        if len(symbols) != len(pos):
            raise ValueError(
                f"Ring {k}: len(symbols)={len(symbols)} != "
                f"len(positions)={len(pos)}; example symbols={symbols[:5]}"
            )

        all_symbols.extend(symbols)
        all_positions.append(pos)

        ring_ranges.append(range(offset, offset + len(symbols)))
        ring_species.append(ring.species)
        offset += len(symbols)

    atoms = Atoms(
        symbols=all_symbols,
        positions=(
            np.vstack(all_positions) if all_positions else np.zeros((0, 3))
        ),
        cell=cell if cell is not None else None,
        pbc=pbc,
    )

    return PackedRings(
        atoms=atoms, ring_ranges=ring_ranges, ring_species=ring_species
    )


def add_ring_bonds_for_lammps_full(
    packed: PackedRings,
    *,
    ring_bond_type: int | Sequence[int] = 1,
    links_ring_local: Iterable[RingLocalLink] = (),
    links_global: Iterable[GlobalLink] = (),
    mol_ids: Sequence[int] | None = None,
    charges: Sequence[float] | None = None,
) -> Atoms:
    """Mutates packed.atoms in-place and returns it."""

    atoms = packed.atoms
    n = len(atoms)

    # ring bond type per ring
    if isinstance(ring_bond_type, int):
        ring_types = [ring_bond_type] * len(packed.ring_ranges)
    else:
        ring_types = list(ring_bond_type)
        if len(ring_types) != len(packed.ring_ranges):
            raise ValueError(
                "ring_bond_type must be int or have one entry per ring"
            )

    neigh: list[list[tuple[int, int]]] = [[] for _ in range(n)]

    for r_range, t in zip(packed.ring_ranges, ring_types):
        ring_globals = list(r_range)
        for i, j in cycle_edges(ring_globals):
            _add_undirected_once(neigh, i, j, int(t))

    for ri, li, rj, lj, bt in links_ring_local:
        if not (
            0 <= ri < len(packed.ring_ranges)
            and 0 <= rj < len(packed.ring_ranges)
        ):
            raise IndexError("link ring index out of range")
        gi = packed.ring_ranges[ri].start + li
        gj = packed.ring_ranges[rj].start + lj
        _add_undirected_once(neigh, gi, gj, int(bt))

    for gi, gj, bt in links_global:
        _add_undirected_once(neigh, int(gi), int(gj), int(bt))

    for a in range(n):
        seen = set()
        uniq = []
        for b, t in neigh[a]:
            key = (b, t)
            if key not in seen:
                seen.add(key)
                uniq.append((b, t))
        neigh[a] = uniq

    atoms.set_array("bonds", adjacency_to_ase_bonds_array(neigh))

    if mol_ids is None:
        mid = np.zeros(n, dtype=int)
        for r_idx, r_range in enumerate(packed.ring_ranges):
            mid[list(r_range)] = r_idx + 1
        atoms.set_array("mol-id", mid)
    else:
        if len(mol_ids) != n:
            raise ValueError("mol_ids must have length == len(atoms)")
        atoms.set_array("mol-id", np.asarray(mol_ids, dtype=int))

    if charges is None:
        atoms.set_initial_charges(np.zeros(n, dtype=float))
    else:
        if len(charges) != n:
            raise ValueError("charges must have length == len(atoms)")
        atoms.set_initial_charges(np.asarray(charges, dtype=float))

    return atoms


def write_rings_lammps_full(
    rings: Sequence,  # Sequence[RingGeometry]
    filepath: str,
    *,
    cell: np.ndarray | None = None,
    pbc: bool | tuple[bool, bool, bool] = False,
    images: np.ndarray | None = None,
    image_convention: str = "subtract",
    ring_bond_type: int | Sequence[int] = 1,
    links_ring_local: Iterable[RingLocalLink] = (),
    links_global: Iterable[GlobalLink] = (),
    masses: bool = True,
) -> PackedRings:
    """
    Returns PackedRings for debugging (atom ordering, ring index ranges).
    """

    packed = pack_rings_to_ase(
        rings,
        cell=cell,
        pbc=pbc,
        images=images,
        image_convention=image_convention,
    )

    add_ring_bonds_for_lammps_full(
        packed,
        ring_bond_type=ring_bond_type,
        links_ring_local=links_ring_local,
        links_global=links_global,
    )

    set_types_by_ring_and_element(packed)

    packed.atoms.write(
        filepath,
        format="lammps-data",
        atom_style="full",
        bonds=True,
        masses=masses,
    )

    return packed
