from __future__ import annotations

from typing import Iterable, Literal, Sequence, overload

import numpy as np
import numpy.typing as npt

from topo_metrics.paths import RingStatistics as RS

# ----------------------------
# Types
# ----------------------------

Float64Array = npt.NDArray[np.float64]
Int64Array = npt.NDArray[np.int64]

PBC3 = tuple[bool, bool, bool]
Shift3 = tuple[int, int, int]


# ----------------------------
# Small helpers
# ----------------------------


def _as_f64_2d(x: npt.ArrayLike, *, shape1: int | None = None) -> Float64Array:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"expected 2D array; got shape {a.shape}")
    if shape1 is not None and a.shape[1] != shape1:
        raise ValueError(f"expected shape (N,{shape1}); got {a.shape}")
    return np.ascontiguousarray(a)


def _as_i64_2d(x: npt.ArrayLike, *, shape1: int | None = None) -> Int64Array:
    a = np.asarray(x, dtype=np.int64)
    if a.ndim != 2:
        raise ValueError(f"expected 2D array; got shape {a.shape}")
    if shape1 is not None and a.shape[1] != shape1:
        raise ValueError(f"expected shape (N,{shape1}); got {a.shape}")
    return np.ascontiguousarray(a)


def _as_cell_3x3(cell: npt.ArrayLike) -> Float64Array:
    C = np.asarray(cell, dtype=np.float64)
    if C.shape != (3, 3):
        raise ValueError(f"cell must have shape (3,3); got {C.shape}")
    return np.ascontiguousarray(C)


def _as_shift3(x) -> Shift3:
    t = tuple(int(v) for v in x)
    if len(t) != 3:
        raise ValueError(f"expected 3-tuple shift; got {t}")
    return t[0], t[1], t[2]


# ----------------------------
# Existing RingStatistics wrappers (typed)
# ----------------------------


def run_rings(
    edges: npt.ArrayLike, depth: int
) -> tuple[Int64Array, list[list[list[tuple[int, Int64Array]]]]]:
    edges_i64 = _as_i64_2d(edges, shape1=5)
    rcount, _, rnodes = RS.run_rings(edges_i64, int(depth))
    rcount_np = np.asarray(rcount, dtype=np.int64)

    converted_rnodes: list[list[list[tuple[int, Int64Array]]]] = [
        [
            [(int(t[0]), np.asarray(t[1], dtype=np.int64)) for t in ring]
            for ring in node
        ]
        for node in rnodes
    ]
    return rcount_np, converted_rnodes


def run_strong_rings(
    edges: npt.ArrayLike, depth: int
) -> tuple[Int64Array, list[list[list[tuple[int, Int64Array]]]]]:
    edges_i64 = _as_i64_2d(edges, shape1=5)
    rcount, _, rnodes = RS.run_strong_rings(edges_i64, int(depth))
    rcount_np = np.asarray(rcount, dtype=np.int64)

    converted_rnodes: list[list[list[tuple[int, Int64Array]]]] = [
        [
            [(int(t[0]), np.asarray(t[1], dtype=np.int64)) for t in ring]
            for ring in node
        ]
        for node in rnodes
    ]
    return rcount_np, converted_rnodes


def get_topological_genome(
    nodes: npt.ArrayLike,
    edges: npt.ArrayLike,
    cell_lengths: Iterable[float],
    cell_angles: Iterable[float],
) -> str:
    X = _as_f64_2d(nodes, shape1=3).T  # 3×N
    edges_i64 = _as_i64_2d(edges, shape1=5)
    return str(
        RS.get_topological_genome(
            X, edges_i64, tuple(cell_lengths), tuple(cell_angles)
        )
    )


def get_all_rings(
    edges: npt.ArrayLike, depth: int
) -> list[list[tuple[int, Int64Array]]]:
    edges_i64 = _as_i64_2d(edges, shape1=5)
    all_rings = RS.get_all_rings(edges_i64, int(depth))
    rings = [
        [(int(t[0]), np.asarray(t[1], dtype=np.int64)) for t in ring]
        for ring in all_rings
    ]
    return rings


def get_coordination_sequences(edges: npt.ArrayLike, dmax: int) -> Int64Array:
    edges_i64 = _as_i64_2d(edges, shape1=5)
    all_coord_seqs = RS.run_coordination_sequences(edges_i64, int(dmax))
    coord_seqs = [[int(t) for t in seq] for seq in all_coord_seqs]
    return np.asarray(coord_seqs, dtype=np.int64)


# ---- bond-distance RDF (with overloads) ----


@overload
def run_bond_distance_rdf(
    positions: npt.ArrayLike,
    edges: npt.ArrayLike,
    cell_lengths: Iterable[float],
    cell_angles: Iterable[float],
    *,
    dmax: int = ...,
    rmax: float = ...,
    dr: float = ...,
    normalise: bool = ...,
    node_types: Sequence[str] | None = ...,
    partial: Literal[False] = ...,
) -> tuple[Float64Array, Float64Array, Float64Array]: ...


@overload
def run_bond_distance_rdf(
    positions: npt.ArrayLike,
    edges: npt.ArrayLike,
    cell_lengths: Iterable[float],
    cell_angles: Iterable[float],
    *,
    dmax: int = ...,
    rmax: float = ...,
    dr: float = ...,
    normalise: bool = ...,
    node_types: Sequence[str] | None = ...,
    partial: Literal[True] = ...,
) -> tuple[Float64Array, list[tuple[str, str]], Float64Array, Float64Array]: ...


def run_bond_distance_rdf(
    positions: npt.ArrayLike,
    edges: npt.ArrayLike,
    cell_lengths: Iterable[float],
    cell_angles: Iterable[float],
    *,
    dmax: int = 6,
    rmax: float = 10.0,
    dr: float = 0.02,
    normalise: bool = True,
    node_types: Sequence[str] | None = None,
    partial: bool = False,
):
    """
    Bond-distance decomposed RDF using the Julia implementation.

    positions: (N,3) Cartesian coordinates.
    edges:     (M,5) [src, dst, ox, oy, oz] with 1-based vertex indices.
    """

    pos = _as_f64_2d(positions, shape1=3)
    edges_i64 = _as_i64_2d(edges, shape1=5)
    X = pos.T  # 3×N

    if not partial:
        r, gD, gtot = RS.run_bond_distance_rdf(
            X,
            edges_i64,
            np.asarray(list(cell_lengths), dtype=np.float64),
            np.asarray(list(cell_angles), dtype=np.float64),
            dmax=int(dmax),
            rmax=float(rmax),
            dr=float(dr),
            normalise=bool(normalise),
            partial=False,
        )
        return (
            np.asarray(r, dtype=np.float64),
            np.asarray(gD, dtype=np.float64),
            np.asarray(gtot, dtype=np.float64),
        )

    # partial=True
    if node_types is None:
        node_types = ["X"] * pos.shape[0]
    else:
        if len(node_types) != pos.shape[0]:
            raise ValueError(
                f"node_types must have length N={pos.shape[0]}; "
                f"got {len(node_types)}"
            )

    r, pairs, gD_pairs, gtot_pairs = RS.run_bond_distance_rdf(
        X,
        edges_i64,
        np.asarray(list(cell_lengths), dtype=np.float64),
        np.asarray(list(cell_angles), dtype=np.float64),
        dmax=int(dmax),
        rmax=float(rmax),
        dr=float(dr),
        normalise=bool(normalise),
        node_types=list(node_types),
        partial=True,
    )
    py_pairs = [(str(a), str(b)) for (a, b) in pairs]

    return (
        np.asarray(r, dtype=np.float64),
        py_pairs,
        np.asarray(gD_pairs, dtype=np.float64),
        np.asarray(gtot_pairs, dtype=np.float64),
    )


# ----------------------------
# NEW: Knots.jl wrappers
# ----------------------------


def linking_number_1a(
    ringA: npt.ArrayLike,
    ringB: npt.ArrayLike,
    *,
    eps: float = 1e-12,
    disjoint_tol: float | None = None,
    disjoint_rel: float = 1e-3,
) -> float:
    """
    Non-PBC Gauss linking number (method 1a) in Julia.

    Rings are N×3 unwrapped Cartesian coordinates.
    Returns NaN if rings are not disjoint (per disjoint_tol / disjoint_rel).
    """
    A = _as_f64_2d(ringA, shape1=3)
    B = _as_f64_2d(ringB, shape1=3)

    lk = RS.Knots.linking_number_1a(
        A,
        B,
        eps=float(eps),
        disjoint_tol=None if disjoint_tol is None else float(disjoint_tol),
        disjoint_rel=float(disjoint_rel),
    )
    return float(lk)


def linking_number_pbc(
    ringA: npt.ArrayLike,
    ringB: npt.ArrayLike,
    cell: npt.ArrayLike,
    *,
    method: Literal["1a"] = "1a",
    pbc: PBC3 = (True, True, True),
    n_images: int = 1,
    eps: float = 1e-12,
    check_top_k: int | None = None,
    disjoint_tol: float | None = None,
    disjoint_rel: float = 1e-3,
    integer_tol: float = 1e-6,
) -> tuple[float, Shift3]:
    """
    Stable wrapper API (keeps a `method=` keyword on the Python side).

    Currently implemented:
      - method="1a" -> RS.Knots.linking_number_pbc_1a

    Rings are N×3 unwrapped Cartesian coordinates.
    """
    if method != "1a":
        raise NotImplementedError(
            "Only method='1a' is currently implemented in Knots.jl"
        )

    A = _as_f64_2d(ringA, shape1=3)
    B = _as_f64_2d(ringB, shape1=3)
    C = _as_cell_3x3(cell)

    lk, sh = RS.Knots.linking_number_pbc_1a(
        A,
        B,
        C,
        pbc=tuple(bool(x) for x in pbc),
        n_images=int(n_images),
        eps=float(eps),
        check_top_k=None if check_top_k is None else int(check_top_k),
        disjoint_tol=None if disjoint_tol is None else float(disjoint_tol),
        disjoint_rel=float(disjoint_rel),
        integer_tol=float(integer_tol),
    )
    return float(lk), _as_shift3(sh)


def linking_number_pbc_1a(
    ringA: npt.ArrayLike,
    ringB: npt.ArrayLike,
    cell: npt.ArrayLike,
    *,
    pbc: PBC3 = (True, True, True),
    n_images: int = 1,
    eps: float = 1e-12,
    check_top_k: int | None = None,
    disjoint_tol: float | None = None,
    disjoint_rel: float = 1e-3,
    integer_tol: float = 1e-6,
) -> tuple[float, Shift3]:
    """
    Direct wrapper for RS.Knots.linking_number_pbc_1a (no `method=` keyword).
    """
    A = _as_f64_2d(ringA, shape1=3)
    B = _as_f64_2d(ringB, shape1=3)
    C = _as_cell_3x3(cell)

    lk, sh = RS.Knots.linking_number_pbc_1a(
        A,
        B,
        C,
        pbc=tuple(bool(x) for x in pbc),
        n_images=int(n_images),
        eps=float(eps),
        check_top_k=None if check_top_k is None else int(check_top_k),
        disjoint_tol=None if disjoint_tol is None else float(disjoint_tol),
        disjoint_rel=float(disjoint_rel),
        integer_tol=float(integer_tol),
    )
    return float(lk), _as_shift3(sh)


def all_pairs_linking_pbc_1a(
    rings: Sequence[npt.ArrayLike],
    cell: npt.ArrayLike,
    *,
    pbc: PBC3 = (True, True, True),
    n_images: int = 1,
    eps: float = 1e-12,
    check_top_k: int | None = None,
    disjoint_tol: float | None = None,
    disjoint_rel: float = 1e-3,
    integer_tol: float = 1e-6,
    zero_based: bool = True,
) -> tuple[Float64Array, Int64Array, Int64Array, Int64Array]:
    """
    Compute all unique pairs i<j using Julia threading.

    Returns:
      lks    : (npairs,) float64
      shifts : (npairs,3) int64   (nx,ny,nz) such that Bimg = B - (cell' * n)
      I      : (npairs,) int64    indices of ring A
      J      : (npairs,) int64    indices of ring B

    By default I,J are converted to 0-based indices for Python.
    """
    C = _as_cell_3x3(cell)
    rings_f = [_as_f64_2d(r, shape1=3) for r in rings]

    lks, shifts, I, J = RS.Knots.all_pairs_linking_pbc_1a(
        rings_f,
        C,
        pbc=tuple(bool(x) for x in pbc),
        n_images=int(n_images),
        eps=float(eps),
        check_top_k=None if check_top_k is None else int(check_top_k),
        disjoint_tol=None if disjoint_tol is None else float(disjoint_tol),
        disjoint_rel=float(disjoint_rel),
        integer_tol=float(integer_tol),
    )

    lks_np = np.asarray(lks, dtype=np.float64)
    shifts_np = np.asarray(shifts, dtype=np.int64)  # should become (npairs, 3)
    I_np = np.asarray(I, dtype=np.int64)
    J_np = np.asarray(J, dtype=np.int64)

    if zero_based:
        I_np = I_np - 1
        J_np = J_np - 1

    # Guard shape for shifts if PythonCall returns list-of-tuples
    if shifts_np.ndim == 1 and shifts_np.size == 3 * lks_np.size:
        shifts_np = shifts_np.reshape((-1, 3))

    return lks_np, shifts_np, I_np, J_np
