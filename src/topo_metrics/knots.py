from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

try:
    import numba as nb  # pyright: ignore[reportMissingImports]
except ImportError:  # pragma: no cover
    nb = None

Array = npt.NDArray[np.floating]


def _drop_duplicate_endpoint(P: Array) -> Array:
    """Drop duplicate last point if it matches the first."""
    if len(P) >= 2 and np.allclose(P[0], P[-1]):
        return P[:-1]
    return P


def _as_points(points: npt.ArrayLike) -> Array:
    """Convert input to (N,3) array of points (drops duplicate endpoint)."""
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("points must have shape (N,3)")
    return _drop_duplicate_endpoint(P)


def _segments(points: Array, closed: bool) -> tuple[Array, Array]:
    """Returns (A, B) where each segment is A[i] -> B[i]."""
    if len(points) < 2:
        return points[:0], points[:0]
    if closed:
        A = points
        B = np.roll(points, -1, axis=0)
    else:
        A = points[:-1]
        B = points[1:]
    return A, B


def _are_adjacent(i: int, j: int, m: int, closed: bool) -> bool:
    """Are segments i and j adjacent (share a vertex)?"""
    if i == j:
        return True
    if closed:
        return (j == (i + 1) % m) or (i == (j + 1) % m)
    return abs(i - j) == 1


def _unit(v: Array, eps: float) -> Array | None:
    """Return unit vector or None if zero norm."""
    n = float(np.linalg.norm(v))
    if n < eps:
        return None
    return v / n


def _cross2(a: Array, b: Array) -> float:
    """2D cross product (scalar)."""
    return float(a[0] * b[1] - a[1] * b[0])


def _as_contig_f64(P: Array) -> np.ndarray:
    """Ensure contiguous float64 array (helps numba)."""
    return np.ascontiguousarray(np.asarray(P, dtype=np.float64))


def _directional_writhe_Wrz(points: Array, *, eps: float = 1e-12) -> float:
    """
    Directional writhe for projection on xy plane (Eq 32).
    Counts segment intersections in 2D and assigns sign via triple product.
    """
    A, B = _segments(points, closed=True)
    S = B - A
    m = len(S)

    Wrz = 0.0
    for i in range(m):
        for j in range(i + 1, m):
            if _are_adjacent(i, j, m, closed=True):
                continue

            p = A[i][:2]
            r = S[i][:2]
            q = A[j][:2]
            s = S[j][:2]

            denom = _cross2(r, s)
            if abs(denom) < eps:
                continue

            qp = q - p
            ti = _cross2(qp, s) / denom
            tj = _cross2(qp, r) / denom

            if not (eps < ti < 1.0 - eps and eps < tj < 1.0 - eps):
                continue

            triple = float(np.dot(np.cross(S[j], S[i]), (A[j] - A[i])))
            if triple > 0:
                Wrz += 1.0
            elif triple < 0:
                Wrz -= 1.0

    return Wrz


def _clean_writhe(wr: float, acn: float, atol: float = 1e-12) -> float:
    """Clean writhe value based on average crossing number."""

    tol = max(atol, 1e-12 * max(1.0, acn))
    return 0.0 if abs(wr) < tol else wr


def _gauss_pair_method_1a(
    p1: Array, p2: Array, p3: Array, p4: Array, eps: float = 1e-12
) -> tuple[float, float]:
    """
    Method 1a: solid-angle / spherical quadrilateral.
    Klenin & Langowski (2000), Eqs (15)-(16).
    """

    def ucr(a, b):
        return _unit(np.cross(a, b), eps)

    r12 = p2 - p1
    r34 = p4 - p3

    r13 = p3 - p1
    r14 = p4 - p1
    r23 = p3 - p2
    r24 = p4 - p2

    n1 = ucr(r13, r14)
    n2 = ucr(r14, r24)
    n3 = ucr(r24, r23)
    n4 = ucr(r23, r13)

    if n1 is None or n2 is None or n3 is None or n4 is None:
        return 0.0, 0.0

    t1 = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
    t2 = float(np.clip(np.dot(n2, n3), -1.0, 1.0))
    t3 = float(np.clip(np.dot(n3, n4), -1.0, 1.0))
    t4 = float(np.clip(np.dot(n4, n1), -1.0, 1.0))

    V_star = math.asin(t1) + math.asin(t2) + math.asin(t3) + math.asin(t4)

    triple = float(np.dot(np.cross(r34, r12), r13))
    sgn = 1.0 if triple > 0 else (-1.0 if triple < 0 else 0.0)

    return V_star * sgn, abs(V_star)


def _gauss_pair_method_1b(
    p1: Array, p2: Array, p3: Array, p4: Array, eps: float = 1e-12
) -> tuple[float, float]:
    """
    Method 1b: analytic evaluation of the Gauss integral.
    Klenin & Langowski (2000), Eqs (17)-(25).
    """

    def F(t1: float, t2: float) -> float:
        rad = t1 * t1 + t2 * t2 - 2.0 * t1 * t2 * cosb + a0 * a0 * sin2b
        if rad <= 0.0:  # pragma: no cover
            return 0.0
        denom = a0 * math.sqrt(rad)
        num = t1 * t2 + a0 * a0 * cosb
        return -(1.0 / (4.0 * math.pi)) * math.atan(num / denom)

    s1 = p2 - p1
    s2 = p4 - p3
    l1 = float(np.linalg.norm(s1))
    l2 = float(np.linalg.norm(s2))
    if l1 < eps or l2 < eps:
        return 0.0, 0.0

    e1 = s1 / l1
    e2 = s2 / l2

    cosb = float(np.dot(e1, e2))
    sin2b = 1.0 - cosb * cosb
    if sin2b < eps:
        return 0.0, 0.0

    r12 = p3 - p1

    a1 = float(np.dot(r12, (e2 * cosb - e1)) / sin2b)
    a2 = float(np.dot(r12, (e2 - e1 * cosb)) / sin2b)
    a0 = float(np.dot(r12, np.cross(e1, e2)) / sin2b)

    if abs(a0) < eps:
        return 0.0, 0.0

    V_over_4pi = (
        F(a1 + l1, a2 + l2) - F(a1 + l1, a2) - F(a1, a2 + l2) + F(a1, a2)
    )

    V = 4.0 * math.pi * V_over_4pi
    return V, abs(V)


if nb is not None:  # pragma: no cover

    @nb.njit(cache=True)
    def _dot3(ax, ay, az, bx, by, bz):
        return ax * bx + ay * by + az * bz

    @nb.njit(cache=True)
    def _cross3(ax, ay, az, bx, by, bz):
        return (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)

    @nb.njit(cache=True)
    def _norm3(x, y, z):
        return math.sqrt(x * x + y * y + z * z)

    @nb.njit(cache=True)
    def _clamp(x, lo, hi):
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x

    @nb.njit(cache=True)
    def _unit_cross(ax, ay, az, bx, by, bz, eps):
        cx, cy, cz = _cross3(ax, ay, az, bx, by, bz)
        n = _norm3(cx, cy, cz)
        if n < eps:
            return 0.0, 0.0, 0.0, False
        inv = 1.0 / n
        return cx * inv, cy * inv, cz * inv, True

    @nb.njit(cache=True)
    def _writhe_sum_1a_numba(A, B, closed_flag, eps):
        """
        Returns (sumV, sumVabs) for method 1a on segments A[i]->B[i].
        Mimics _gauss_pair_method_1a logic and adjacency skipping.
        """
        m = A.shape[0]
        sumV = 0.0
        sumVabs = 0.0

        for i in range(m):
            p1x, p1y, p1z = A[i, 0], A[i, 1], A[i, 2]
            p2x, p2y, p2z = B[i, 0], B[i, 1], B[i, 2]
            r12x, r12y, r12z = p2x - p1x, p2y - p1y, p2z - p1z

            for j in range(i + 1, m):
                if closed_flag == 1:
                    if j == i + 1:
                        continue
                    if i == 0 and j == m - 1:
                        continue
                else:
                    if j == i + 1:
                        continue

                p3x, p3y, p3z = A[j, 0], A[j, 1], A[j, 2]
                p4x, p4y, p4z = B[j, 0], B[j, 1], B[j, 2]
                r34x, r34y, r34z = p4x - p3x, p4y - p3y, p4z - p3z

                r13x, r13y, r13z = p3x - p1x, p3y - p1y, p3z - p1z
                r14x, r14y, r14z = p4x - p1x, p4y - p1y, p4z - p1z
                r23x, r23y, r23z = p3x - p2x, p3y - p2y, p3z - p2z
                r24x, r24y, r24z = p4x - p2x, p4y - p2y, p4z - p2z

                n1x, n1y, n1z, ok1 = _unit_cross(
                    r13x, r13y, r13z, r14x, r14y, r14z, eps
                )
                if not ok1:
                    continue
                n2x, n2y, n2z, ok2 = _unit_cross(
                    r14x, r14y, r14z, r24x, r24y, r24z, eps
                )
                if not ok2:
                    continue
                n3x, n3y, n3z, ok3 = _unit_cross(
                    r24x, r24y, r24z, r23x, r23y, r23z, eps
                )
                if not ok3:
                    continue
                n4x, n4y, n4z, ok4 = _unit_cross(
                    r23x, r23y, r23z, r13x, r13y, r13z, eps
                )
                if not ok4:
                    continue

                t1 = _clamp(_dot3(n1x, n1y, n1z, n2x, n2y, n2z), -1.0, 1.0)
                t2 = _clamp(_dot3(n2x, n2y, n2z, n3x, n3y, n3z), -1.0, 1.0)
                t3 = _clamp(_dot3(n3x, n3y, n3z, n4x, n4y, n4z), -1.0, 1.0)
                t4 = _clamp(_dot3(n4x, n4y, n4z, n1x, n1y, n1z), -1.0, 1.0)

                V_star = (
                    math.asin(t1)
                    + math.asin(t2)
                    + math.asin(t3)
                    + math.asin(t4)
                )
                V_abs = -V_star if V_star < 0.0 else V_star
                sumVabs += V_abs

                cx, cy, cz = _cross3(r34x, r34y, r34z, r12x, r12y, r12z)
                triple = _dot3(cx, cy, cz, r13x, r13y, r13z)

                if triple > 0.0:
                    sumV += V_star
                elif triple < 0.0:
                    sumV -= V_star

        return sumV, sumVabs

    @nb.njit(cache=True)
    def _link_sum_1a_numba(A1, B1, A2, B2, sx, sy, sz, eps):
        """
        Returns sumV (signed solid-angle sum) for linking between two rings,
        with ring2 shifted by (sx,sy,sz) applied as: p -> p - shift.
        """
        sumV = 0.0

        for i in range(A1.shape[0]):
            p1x, p1y, p1z = A1[i, 0], A1[i, 1], A1[i, 2]
            p2x, p2y, p2z = B1[i, 0], B1[i, 1], B1[i, 2]
            r12x, r12y, r12z = p2x - p1x, p2y - p1y, p2z - p1z

            for j in range(A2.shape[0]):
                p3x, p3y, p3z = A2[j, 0] - sx, A2[j, 1] - sy, A2[j, 2] - sz
                p4x, p4y, p4z = B2[j, 0] - sx, B2[j, 1] - sy, B2[j, 2] - sz

                r34x, r34y, r34z = p4x - p3x, p4y - p3y, p4z - p3z

                r13x, r13y, r13z = p3x - p1x, p3y - p1y, p3z - p1z
                r14x, r14y, r14z = p4x - p1x, p4y - p1y, p4z - p1z
                r23x, r23y, r23z = p3x - p2x, p3y - p2y, p3z - p2z
                r24x, r24y, r24z = p4x - p2x, p4y - p2y, p4z - p2z

                n1x, n1y, n1z, ok1 = _unit_cross(
                    r13x, r13y, r13z, r14x, r14y, r14z, eps
                )
                if not ok1:
                    continue
                n2x, n2y, n2z, ok2 = _unit_cross(
                    r14x, r14y, r14z, r24x, r24y, r24z, eps
                )
                if not ok2:
                    continue
                n3x, n3y, n3z, ok3 = _unit_cross(
                    r24x, r24y, r24z, r23x, r23y, r23z, eps
                )
                if not ok3:
                    continue
                n4x, n4y, n4z, ok4 = _unit_cross(
                    r23x, r23y, r23z, r13x, r13y, r13z, eps
                )
                if not ok4:
                    continue

                t1 = _clamp(_dot3(n1x, n1y, n1z, n2x, n2y, n2z), -1.0, 1.0)
                t2 = _clamp(_dot3(n2x, n2y, n2z, n3x, n3y, n3z), -1.0, 1.0)
                t3 = _clamp(_dot3(n3x, n3y, n3z, n4x, n4y, n4z), -1.0, 1.0)
                t4 = _clamp(_dot3(n4x, n4y, n4z, n1x, n1y, n1z), -1.0, 1.0)

                V_star = (
                    math.asin(t1)
                    + math.asin(t2)
                    + math.asin(t3)
                    + math.asin(t4)
                )

                cx, cy, cz = _cross3(r34x, r34y, r34z, r12x, r12y, r12z)
                triple = _dot3(cx, cy, cz, r13x, r13y, r13z)

                if triple > 0.0:
                    sumV += V_star
                elif triple < 0.0:
                    sumV -= V_star

        return sumV

    # --- segment-segment distance (numba) for disjointness & ranking ---

    @nb.njit(cache=True)
    def _segseg_dist2_numba(
        p1x, p1y, p1z, p2x, p2y, p2z, q1x, q1y, q1z, q2x, q2y, q2z
    ):
        """
        Ericson-style segment-segment distance squared, numerically stable.
        """

        ux, uy, uz = p2x - p1x, p2y - p1y, p2z - p1z
        vx, vy, vz = q2x - q1x, q2y - q1y, q2z - q1z
        wx, wy, wz = p1x - q1x, p1y - q1y, p1z - q1z

        a = ux * ux + uy * uy + uz * uz
        b = ux * vx + uy * vy + uz * vz
        c = vx * vx + vy * vy + vz * vz
        d = ux * wx + uy * wy + uz * wz
        e = vx * wx + vy * wy + vz * wz
        D = a * c - b * b

        sN, sD = 0.0, D
        tN, tD = 0.0, D

        eps = 1e-15
        if D < eps:
            sN, sD = 0.0, 1.0
            tN, tD = e, c
        else:
            sN = b * e - c * d
            tN = a * e - b * d

            if sN < 0.0:
                sN = 0.0
                tN = e
                tD = c
            elif sN > sD:
                sN = sD
                tN = e + b
                tD = c

        if tN < 0.0:
            tN = 0.0
            if -d < 0.0:
                sN = 0.0
            elif -d > a:
                sN = sD
            else:
                sN = -d
                sD = a
        elif tN > tD:
            tN = tD
            if (-d + b) < 0.0:
                sN = 0.0
            elif (-d + b) > a:
                sN = sD
            else:
                sN = -d + b
                sD = a

        sc = 0.0 if abs(sN) < eps else (sN / sD)
        tc = 0.0 if abs(tN) < eps else (tN / tD)

        dx = wx + sc * ux - tc * vx
        dy = wy + sc * uy - tc * vy
        dz = wz + sc * uz - tc * vz
        return dx * dx + dy * dy + dz * dz

    @nb.njit(cache=True)
    def _min_seg_dist2_shift_numba(A1, B1, A2, B2, sx, sy, sz, early_exit2):
        """
        Min squared distance between segments A1->B1 and shifted A2->B2.
        Shift applied as: q -> q - (sx,sy,sz).
        If early_exit2 >= 0 and best <= early_exit2, returns early.
        """
        best = 1.0e300
        for i in range(A1.shape[0]):
            p1x, p1y, p1z = A1[i, 0], A1[i, 1], A1[i, 2]
            p2x, p2y, p2z = B1[i, 0], B1[i, 1], B1[i, 2]
            for j in range(A2.shape[0]):
                q1x, q1y, q1z = A2[j, 0] - sx, A2[j, 1] - sy, A2[j, 2] - sz
                q2x, q2y, q2z = B2[j, 0] - sx, B2[j, 1] - sy, B2[j, 2] - sz
                d2 = _segseg_dist2_numba(
                    p1x, p1y, p1z, p2x, p2y, p2z, q1x, q1y, q1z, q2x, q2y, q2z
                )
                if d2 < best:
                    best = d2
                    if early_exit2 >= 0.0 and best <= early_exit2:
                        return best
        return best


def writhe_method_1a(
    points: npt.ArrayLike, *, closed: bool = True, eps: float = 1e-12
) -> tuple[float, float]:
    """Method 1a: pairwise solid angles (Eqs 13, 15-16)."""

    P = _as_points(points)
    A, B = _segments(P, closed)
    m = len(A)
    if m < 2:
        return 0.0, 0.0

    if nb is not None:
        Af = _as_contig_f64(A)
        Bf = _as_contig_f64(B)
        sumV, sumVabs = _writhe_sum_1a_numba(Af, Bf, 1 if closed else 0, eps)
    else:
        sumV = 0.0
        sumVabs = 0.0
        for i in range(m):
            for j in range(i + 1, m):
                if _are_adjacent(i, j, m, closed):
                    continue
                V, Vabs = _gauss_pair_method_1a(A[i], B[i], A[j], B[j], eps=eps)
                sumV += V
                sumVabs += Vabs

    pref = 1.0 / (2.0 * math.pi)
    writhe = _clean_writhe(sumV * pref, sumVabs * pref, atol=eps)
    acn = sumVabs * pref
    return float(writhe), float(acn)


def writhe_method_1b(
    points: npt.ArrayLike, *, closed: bool = True, eps: float = 1e-12
) -> tuple[float, float]:
    """Method 1b: analytic Gauss integral (Eqs 13, 24-25)."""
    P = _as_points(points)
    A, B = _segments(P, closed)
    m = len(A)
    if m < 2:
        return 0.0, 0.0

    sumV = 0.0
    sumVabs = 0.0
    for i in range(m):
        for j in range(i + 1, m):
            if _are_adjacent(i, j, m, closed):
                continue
            V, Vabs = _gauss_pair_method_1b(A[i], B[i], A[j], B[j], eps=eps)
            sumV += V
            sumVabs += Vabs

    pref = 1.0 / (2.0 * math.pi)
    writhe = _clean_writhe(sumV * pref, sumVabs * pref, atol=eps)
    acn = sumVabs * pref
    return float(writhe), float(acn)


def writhe_method_2a(points: npt.ArrayLike, *, eps: float = 1e-12) -> float:
    """
    Method 2a: Wr = Twz + Wrz - Tw (Eqs 30-34), z-axis projection.
    Requires a closed chain.
    """
    P = _as_points(points)
    n = len(P)
    if n < 4:
        return 0.0

    A, B = _segments(P, closed=True)
    S = B - A

    p = np.zeros((n, 3), dtype=float)
    for i in range(n):
        c = np.cross(S[i - 1], S[i])
        u = _unit(c, eps)
        if u is None:
            return float("nan")
        p[i] = u

    Tw_sum = 0.0
    for i in range(n):
        pi = p[i]
        pip = p[(i + 1) % n]
        ang = math.acos(float(np.clip(np.dot(pi, pip), -1.0, 1.0)))
        sgn = float(np.sign(np.dot(pi, S[(i + 1) % n])))
        Tw_sum += ang * sgn
    Tw = Tw_sum / (2.0 * math.pi)

    Twz = 0.0
    for i in range(n):
        if p[i, 2] * p[(i + 1) % n, 2] < 0.0:
            Twz += float(np.sign(np.dot(p[i], S[(i + 1) % n])))
    Twz *= 0.5

    Wrz = _directional_writhe_Wrz(P, eps=eps)
    return float(Twz + Wrz - Tw)


def writhe_method_2b(points: npt.ArrayLike, *, eps: float = 1e-12) -> float:
    """
    Method 2b (le Bret-style):
        Wr = Wrz - Tw with a_i = k×s_i/|k×s_i| (Eqs 35-38), k = z-axis.
    Requires closed chain.
    """

    P = _as_points(points)
    n = len(P)
    if n < 4:
        return 0.0

    A, B = _segments(P, closed=True)
    S = B - A

    k = np.array([0.0, 0.0, 1.0], dtype=float)

    a = np.zeros((n, 3), dtype=float)
    for i in range(n):
        u = _unit(np.cross(k, S[i]), eps)
        if u is None:
            return float("nan")
        a[i] = u

    p = np.zeros((n, 3), dtype=float)
    for i in range(n):
        u = _unit(np.cross(S[i - 1], S[i]), eps)
        if u is None:
            return float("nan")
        p[i] = u

    Tw_sum = 0.0
    for i in range(n):
        pi = p[i]
        term1 = math.acos(float(np.clip(np.dot(a[i - 1], pi), -1.0, 1.0)))
        term2 = math.acos(float(np.clip(np.dot(pi, a[i]), -1.0, 1.0)))
        Tw_sum += (term1 - term2) * float(np.sign(pi[2]))
    Tw = Tw_sum / (2.0 * math.pi)

    Wrz = _directional_writhe_Wrz(P, eps=eps)
    return float(Wrz - Tw)


def _median_segment_length(points: Array, closed: bool = True) -> float:
    P = _as_points(points)
    A, B = _segments(P, closed=closed)
    if len(A) == 0:
        return 0.0
    seg = np.linalg.norm(B - A, axis=1)
    return float(np.median(seg))


def _auto_disjoint_tol(
    ring1: Array,
    ring2: Array,
    *,
    rel: float = 1e-3,
    abs_: float = 1e-8,
) -> float:
    """
    Heuristic: disjoint_tol ~ rel * (typical segment length), floored by abs_.

    rel=1e-3: "numerical disjointness" default.
    rel=1e-2: more conservative if you see near-singular Gauss contributions.
    """
    m1 = _median_segment_length(ring1, closed=True)
    m2 = _median_segment_length(ring2, closed=True)
    m = min(m1, m2) if (m1 > 0 and m2 > 0) else max(m1, m2)
    if m <= 0:
        return float(abs_)
    return float(max(abs_, rel * m))


def _segseg_dist2(
    p1: Array, p2: Array, q1: Array, q2: Array, eps: float = 1e-15
) -> float:
    """
    Squared minimum distance between 3D segments p1->p2 and q1->q2.
    Robust clamp-based implementation (Ericson-style).
    """
    u = p2 - p1
    v = q2 - q1
    w = p1 - q1

    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))
    D = a * c - b * b

    sN, sD = 0.0, D
    tN, tD = 0.0, D

    if D < eps:
        sN, sD = 0.0, 1.0
        tN, tD = e, c
    else:
        sN = b * e - c * d
        tN = a * e - b * d

        if sN < 0.0:
            sN = 0.0
            tN = e
            tD = c
        elif sN > sD:
            sN = sD
            tN = e + b
            tD = c

    if tN < 0.0:
        tN = 0.0
        if -d < 0.0:
            sN = 0.0
        elif -d > a:
            sN = sD
        else:
            sN = -d
            sD = a
    elif tN > tD:
        tN = tD
        if (-d + b) < 0.0:
            sN = 0.0
        elif (-d + b) > a:
            sN = sD
        else:
            sN = -d + b
            sD = a

    sc = 0.0 if abs(sN) < eps else (sN / sD)
    tc = 0.0 if abs(tN) < eps else (tN / tD)

    dP = w + sc * u - tc * v
    return float(np.dot(dP, dP))


def _min_seg_dist2(
    A1: Array,
    B1: Array,
    A2: Array,
    B2: Array,
    *,
    early_exit2: float | None = None,
) -> float:
    """Min squared distance between any segment in ring1 and ring2."""
    best = float("inf")
    for i in range(A1.shape[0]):
        p1 = A1[i]
        p2 = B1[i]
        for j in range(A2.shape[0]):
            q1 = A2[j]
            q2 = B2[j]
            d2 = _segseg_dist2(p1, p2, q1, q2)
            if d2 < best:
                best = d2
                if early_exit2 is not None and best <= early_exit2:
                    return float(best)
    return float(best)


def linking_number_method_1a(
    ring1: npt.ArrayLike,
    ring2: npt.ArrayLike,
    *,
    eps: float = 1e-12,
    disjoint_tol: float | None = None,
    disjoint_rel: float = 1e-3,
    return_nan_if_not_disjoint: bool = True,
) -> float:
    """
    Gauss linking number for two CLOSED polygonal rings (method 1a).

    Disjointness:
      - if disjoint_tol is None: uses auto disjoint_tol from disjoint_rel
      - if disjoint_tol <= 0: skips the disjointness check
      - if not disjoint: returns nan or raises return_nan_if_not_disjoint=False
    """

    P = _as_points(ring1)
    Q = _as_points(ring2)

    A1, B1 = _segments(P, closed=True)
    A2, B2 = _segments(Q, closed=True)

    if len(A1) < 2 or len(A2) < 2:
        return 0.0

    if disjoint_tol is None:
        disjoint_tol = _auto_disjoint_tol(P, Q, rel=disjoint_rel)
    if disjoint_tol is not None and disjoint_tol > 0.0:
        d2 = _min_seg_dist2(
            A1, B1, A2, B2, early_exit2=float(disjoint_tol) ** 2
        )
        if d2 < float(disjoint_tol) ** 2:
            if return_nan_if_not_disjoint:
                return float("nan")
            raise ValueError(
                "Rings are not disjoint (segment distance below disjoint_tol)."
            )

    if nb is not None:
        A1f, B1f = _as_contig_f64(A1), _as_contig_f64(B1)
        A2f, B2f = _as_contig_f64(A2), _as_contig_f64(B2)
        sumV = _link_sum_1a_numba(A1f, B1f, A2f, B2f, 0.0, 0.0, 0.0, eps)
        return float(sumV / (4.0 * math.pi))

    sumV = 0.0
    for i in range(len(A1)):
        for j in range(len(A2)):
            V, _ = _gauss_pair_method_1a(A1[i], B1[i], A2[j], B2[j], eps=eps)
            sumV += V
    return float(sumV / (4.0 * math.pi))


def linking_number_method_1b(
    ring1: npt.ArrayLike,
    ring2: npt.ArrayLike,
    *,
    eps: float = 1e-12,
    disjoint_tol: float | None = None,
    disjoint_rel: float = 1e-3,
    return_nan_if_not_disjoint: bool = True,
) -> float:
    """
    Gauss linking number for two CLOSED polygonal rings (method 1b analytic).

    Same disjointness behavior as method_1a.
    """
    P = _as_points(ring1)
    Q = _as_points(ring2)

    A1, B1 = _segments(P, closed=True)
    A2, B2 = _segments(Q, closed=True)

    if len(A1) < 2 or len(A2) < 2:
        return 0.0

    if disjoint_tol is None:
        disjoint_tol = _auto_disjoint_tol(P, Q, rel=disjoint_rel)
    if disjoint_tol is not None and disjoint_tol > 0.0:
        d2 = _min_seg_dist2(
            A1, B1, A2, B2, early_exit2=float(disjoint_tol) ** 2
        )
        if d2 < float(disjoint_tol) ** 2:
            if return_nan_if_not_disjoint:
                return float("nan")
            raise ValueError(
                "Rings are not disjoint (segment distance below disjoint_tol)."
            )

    sumV = 0.0
    for i in range(len(A1)):
        for j in range(len(A2)):
            V, _ = _gauss_pair_method_1b(A1[i], B1[i], A2[j], B2[j], eps=eps)
            sumV += V
    return float(sumV / (4.0 * math.pi))


def lk_round(lk: float, tol: float = 1e-6) -> tuple[int, bool]:
    """Return (rounded_int, ok) where ok means |lk-round(lk)| <= tol."""
    r = int(round(lk))
    return r, (abs(lk - r) <= tol)


def linking_number_int(lk: float, tol: float = 1e-6) -> int:
    """Convert near-integer lk to integer."""

    return lk_round(lk, tol=tol)[0]


def is_linked_from_lk(lk: float, *, tol: float = 1e-6) -> bool:
    """Determine if two rings are linked from linking number value."""

    r, ok = lk_round(lk, tol=tol)
    return bool(ok and abs(r) > 0)


def _centroid(P: Array) -> Array:
    """Compute centroid of points P."""

    return np.mean(P, axis=0)


def _pbc_base_integer_shift(
    cA: Array,
    cB: Array,
    cell: Array,
    pbc: tuple[bool, bool, bool],
) -> Array:
    """
    Returns integer vector n0 such that shifting B by -(n0 @ cell) brings
    centroid difference into the minimum image (fractional in [-0.5, 0.5)).
    """
    inv_cell = np.linalg.inv(cell)
    df = (cB - cA) @ inv_cell

    n0 = np.zeros(3, dtype=int)
    for k in range(3):
        n0[k] = int(np.round(df[k])) if pbc[k] else 0
    return n0


def linking_number_pbc(
    ringA: Array,
    ringB: Array,
    *,
    cell: Array,
    pbc: tuple[bool, bool, bool] = (True, True, True),
    n_images: int = 1,
    method: str = "1a",
    eps: float = 1e-12,
    check_top_k: int | None = None,
    disjoint_tol: float | None = None,
    disjoint_rel: float = 1e-3,
) -> tuple[float, tuple[int, int, int]]:
    """
    Compute Gauss linking number between ringA and ringB under PBC by scanning
    periodic images of ringB.

    Returns (best_lk, best_image_shift) where best_image_shift is integer n such
    that:

        ringB_shifted = ringB - (n @ cell)

    Candidate scoring / selection:
      - compute min segment-segment dist^2 for each shift
      - discard candidates with dist < disjoint_tol
      - among remaining, choose the shift that maximizes |round(lk)|,
        tie-break by smaller dist^2, then smaller residual |lk-round(lk)|
    """

    cell = np.asarray(cell, dtype=float)
    if cell.shape != (3, 3):
        raise ValueError("cell must have shape (3,3)")

    A = np.asarray(ringA, dtype=float)
    B = np.asarray(ringB, dtype=float)
    if A.ndim != 2 or A.shape[1] != 3 or B.ndim != 2 or B.shape[1] != 3:
        raise ValueError("rings must be arrays of shape (N,3)")

    if disjoint_tol is None:
        disjoint_tol = _auto_disjoint_tol(A, B, rel=disjoint_rel)
    disjoint_tol2 = (
        float(disjoint_tol) ** 2
        if (disjoint_tol is not None and disjoint_tol > 0.0)
        else -1.0
    )

    # Base integer shift from centroid minimum image
    n0 = _pbc_base_integer_shift(_centroid(A), _centroid(B), cell, pbc)
    rng = range(-n_images, n_images + 1)

    # Precompute segments
    P = _as_points(A)
    Q = _as_points(B)
    A1, B1 = _segments(P, closed=True)
    A2, B2 = _segments(Q, closed=True)
    if len(A1) < 2 or len(A2) < 2:
        return 0.0, (0, 0, 0)

    # --- Numba accelerated path for method 1a (no B copies) ---
    if nb is not None and method == "1a":
        A1f, B1f = _as_contig_f64(A1), _as_contig_f64(B1)
        A2f, B2f = _as_contig_f64(A2), _as_contig_f64(B2)

        candidates: list[
            tuple[float, tuple[int, int, int], tuple[float, float, float]]
        ] = []
        for i in rng:
            for j in rng:
                for k in rng:
                    n = np.array([n0[0] + i, n0[1] + j, n0[2] + k], dtype=int)
                    for dim in range(3):
                        if not pbc[dim]:
                            n[dim] = 0

                    shift = n @ cell
                    sx, sy, sz = (
                        float(shift[0]),
                        float(shift[1]),
                        float(shift[2]),
                    )

                    dist2 = float(
                        _min_seg_dist2_shift_numba(
                            A1f, B1f, A2f, B2f, sx, sy, sz, disjoint_tol2
                        )
                    )
                    candidates.append(
                        (dist2, (int(n[0]), int(n[1]), int(n[2])), (sx, sy, sz))
                    )

        candidates.sort(key=lambda x: x[0])
        if check_top_k is not None:
            candidates = candidates[: max(1, int(check_top_k))]

        best_lk = float("nan")
        best_n = (0, 0, 0)
        best_int = 0
        best_dist2 = float("inf")
        best_resid = float("inf")

        for dist2, n, (sx, sy, sz) in candidates:
            if disjoint_tol2 >= 0.0 and dist2 < disjoint_tol2:
                continue

            sumV = float(
                _link_sum_1a_numba(A1f, B1f, A2f, B2f, sx, sy, sz, eps)
            )
            lk = sumV / (4.0 * math.pi)

            lk_int = int(round(lk))
            resid = abs(lk - lk_int)

            if (
                abs(lk_int) > abs(best_int)
                or (abs(lk_int) == abs(best_int) and dist2 < best_dist2)
                or (
                    abs(lk_int) == abs(best_int)
                    and dist2 == best_dist2
                    and resid < best_resid
                )
            ):
                best_lk = float(lk)
                best_n = n
                best_int = lk_int
                best_dist2 = dist2
                best_resid = resid

        return best_lk, best_n

    # --- Fallback path (python): build shifted copies ---
    candidates2: list[tuple[float, tuple[int, int, int], Array]] = []
    for i in rng:
        for j in rng:
            for k in rng:
                n = np.array([n0[0] + i, n0[1] + j, n0[2] + k], dtype=int)
                for dim in range(3):
                    if not pbc[dim]:
                        n[dim] = 0

                shift = n @ cell
                Bimg = B - shift

                Qimg = _as_points(Bimg)
                A2i, B2i = _segments(Qimg, closed=True)
                dist2 = _min_seg_dist2(
                    A1,
                    B1,
                    A2i,
                    B2i,
                    early_exit2=disjoint_tol2 if disjoint_tol2 >= 0.0 else None,
                )

                candidates2.append(
                    (dist2, (int(n[0]), int(n[1]), int(n[2])), Bimg)
                )

    candidates2.sort(key=lambda x: x[0])
    if check_top_k is not None:
        candidates2 = candidates2[: max(1, int(check_top_k))]

    best_lk = float("nan")
    best_n = (0, 0, 0)
    best_int = 0
    best_dist2 = float("inf")
    best_resid = float("inf")

    for dist2, n, Bimg in candidates2:
        if disjoint_tol2 >= 0.0 and dist2 < disjoint_tol2:
            continue

        if method == "1a":
            # already filtered by disjointness above; skip redundant check here
            lk = linking_number_method_1a(A, Bimg, eps=eps, disjoint_tol=0.0)
        elif method == "1b":
            lk = linking_number_method_1b(A, Bimg, eps=eps, disjoint_tol=0.0)
        else:
            raise ValueError("method must be '1a' or '1b' for linking number")

        if not np.isfinite(lk):
            continue

        lk_int = int(round(lk))
        resid = abs(lk - lk_int)

        if (
            abs(lk_int) > abs(best_int)
            or (abs(lk_int) == abs(best_int) and dist2 < best_dist2)
            or (
                abs(lk_int) == abs(best_int)
                and dist2 == best_dist2
                and resid < best_resid
            )
        ):
            best_lk = float(lk)
            best_n = n
            best_int = lk_int
            best_dist2 = dist2
            best_resid = resid

    return best_lk, best_n
