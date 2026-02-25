from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import ase
import numpy as np
import numpy.typing as npt
from ase.io import read as ase_read

from topo_metrics.neighbours import (
    autoreduce_neighborlist,
    graph_edges_by_cutoff,
)


@dataclass(frozen=True)
class LammpsDataComponents:
    """Components extracted from a LAMMPS data file."""

    ase_atoms: ase.Atoms
    """ASE atoms object with cell/pbc/positions populated."""

    edges: npt.NDArray[np.int_]
    """Neighbourlist with image shifts: [i, j, sx, sy, sz] (1-based indices)."""

    symbols: list[str]
    """Per-node symbols/types used for Topology node types."""

    cart_coords: npt.NDArray[np.floating]
    """Cartesian coordinates, shape (N, 3)."""

    frac_coords: npt.NDArray[np.floating] | list[None]
    """Fractional coordinates, shape (N, 3) if periodic, else list[None]."""


def load_lammps_data(
    filename: Path | str,
    *,
    atom_style: str = "atomic",
    units: str = "metal",
    sort_by_id: bool = True,
    prefer_bonds: bool = True,
    cutoff: float = 0.0,
    pair_cutoffs: dict[tuple[str, str], float] | None = None,
    contract_neighborlist: bool = False,
    remove_types: Iterable[object] | None = None,
    remove_degree2: bool = False,
    omit_node_types: bool = False,
) -> LammpsDataComponents:
    """Load a LAMMPS data file into ASE plus a periodic bond graph.

    This reader uses ASE's LAMMPS-data support to obtain an ``ase.Atoms`` 
    object. If a ``Bonds`` section is present, bonds are used to construct the 
    neighbourlist; otherwise, a neighbour list can be inferred by cutoff using
    :func:`topo_metrics.neighbours.graph_edges_by_cutoff`.

    The neighbour list uses the convention ``[i, j, sx, sy, sz]`` where indices
    are 1-based and the image shift ``(sx, sy, sz)`` is chosen assuming the
    minimum image convention (MIC).
    """

    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"File '{path}' not found")

    atoms = ase_read(
        path,
        format="lammps-data",
        atom_style=atom_style,
        units=units,
        sort_by_id=sort_by_id,
    )
    assert isinstance(atoms, ase.Atoms)

    type_to_symbol = _parse_masses_symbols(path)
    if type_to_symbol is not None and "type" in atoms.arrays:
        atypes = atoms.arrays["type"].astype(int)
        new_symbols: list[str] = []
        for idx, t in enumerate(atypes):
            sym = type_to_symbol.get(int(t))
            if sym is None:
                sym = atoms.get_chemical_symbols()[idx]
            new_symbols.append(sym)
        atoms.set_chemical_symbols(new_symbols)

    cart_coords = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()

    frac_coords: npt.NDArray[np.floating] | list[None]
    if all(atoms.pbc):
        frac_coords = atoms.get_scaled_positions()
    else:
        frac_coords = [None] * len(atoms)

    edges: npt.NDArray[np.int_]
    bonds = _parse_bonds(path) if prefer_bonds else []
    if bonds:
        edges = _bonds_to_edges_mic(atoms, bonds)
    else:
        edges = graph_edges_by_cutoff(
            atoms, cutoff=cutoff, pair_cutoffs=pair_cutoffs, one_based=True
        )

    if contract_neighborlist or remove_types is not None or remove_degree2:
        reduced = autoreduce_neighborlist(
            cart_coords=cart_coords,
            frac_coords=np.asarray(frac_coords, dtype=np.float64),
            symbols=symbols,
            edges=edges,
            remove_types=remove_types,
            remove_degree2=remove_degree2,
        )
        cart_coords, frac_coords, symbols, edges, _ = reduced

    if omit_node_types:
        symbols = ["X"] * len(symbols)

    return LammpsDataComponents(
        ase_atoms=atoms,
        edges=np.asarray(edges, dtype=int),
        symbols=list(symbols),
        cart_coords=np.asarray(cart_coords, dtype=float),
        frac_coords=frac_coords,
    )


def _parse_masses_symbols(path: Path) -> dict[int, str] | None:
    """Parse `Masses` section extracting inline symbols from ``#`` comments."""

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip() == "Masses")
    except StopIteration:
        return None

    out: dict[int, str] = {}
    i = start + 1

    while i < len(lines) and not lines[i].strip():
        i += 1

    for ln in lines[i:]:
        s = ln.strip()
        if not s:
            break
        if s.split(maxsplit=1)[0].isalpha() and len(s.split()) == 1:
            break

        head, *_ = s.split("#", maxsplit=1)
        parts = head.split()
        if not parts:
            continue
        try:
            t = int(parts[0])
        except ValueError:
            continue

        if "#" in s:
            comment = s.split("#", maxsplit=1)[1].strip()
            sym = comment.split()[0] if comment else ""
            if sym:
                out[t] = sym

    return out or None


def _parse_bonds(path: Path) -> list[tuple[int, int, int]]:
    """Parse the ``Bonds`` section.

    Returns a list of ``(i, j, bond_type)`` where ``i`` and ``j`` are 1-based
    atom IDs from the data file.
    """

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip() == "Bonds")
    except StopIteration:
        return []

    i = start + 1
    while i < len(lines) and not lines[i].strip():
        i += 1

    out: list[tuple[int, int, int]] = []
    for ln in lines[i:]:
        s = ln.strip()
        if not s:
            break
        if s.split(maxsplit=1)[0].isalpha() and len(s.split()) == 1:
            break

        head = s.split("#", maxsplit=1)[0].strip()
        parts = head.split()
        if len(parts) < 4:
            continue
        # bond-id, bond-type, ai, aj
        try:
            btype = int(parts[1])
            ai = int(parts[2])
            aj = int(parts[3])
        except ValueError:
            continue
        out.append((ai, aj, btype))
    return out


def _bonds_to_edges_mic(
    atoms: ase.Atoms, bonds: list[tuple[int, int, int]]
) -> npt.NDArray[np.int_]:
    """Convert LAMMPS bonds into a periodic neighbourlist assuming MIC."""

    if not all(atoms.pbc):
        return np.array([[i, j, 0, 0, 0] for i, j, _ in bonds], dtype=int)

    frac = atoms.get_scaled_positions(wrap=True)

    edges: list[list[int]] = []
    for i, j, _btype in bonds:
        i0 = i - 1
        j0 = j - 1
        if i0 < 0 or j0 < 0 or i0 >= len(frac) or j0 >= len(frac):
            raise ValueError(
                f"Bond references out-of-range atom id: ({i}, {j}) "
                f"for N={len(frac)}"
            )

        df = frac[j0] - frac[i0]
        shift = _mic_shift(df)
        edges.append([i, j, int(shift[0]), int(shift[1]), int(shift[2])])

    return np.asarray(edges, dtype=int)


def _mic_shift(dfrac: npt.NDArray[np.floating]) -> npt.NDArray[np.int_]:
    """Return integer shift to apply to the second atom to satisfy MIC."""

    return (-np.rint(dfrac + 1e-12)).astype(int)
