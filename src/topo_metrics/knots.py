from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.floating]


def _drop_duplicate_endpoint(P: Array) -> Array:
    """Drop duplicate last point if it matches the first."""

    if len(P) >= 2 and np.allclose(P[0], P[-1]):
        return P[:-1]

    return P


def _as_points(points: npt.ArrayLike) -> Array:
    """Convert input to (N,3) array of points."""

    P = np.asarray(points, dtype=float)

    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("points must have shape (N,3)")

    P = _drop_duplicate_endpoint(P)

    return P


def _segments(points: Array, closed: bool) -> tuple[Array, Array]:
    """Returns (A, B) where each segment is A[i] -> B[i]."""

    if closed:
        A = points
        B = np.roll(points, -1, axis=0)
    else:
        if len(points) < 2:
            return points[:0], points[:0]
        A = points[:-1]
        B = points[1:]

    return A, B


def _are_adjacent(i: int, j: int, m: int, closed: bool) -> bool:
    """Are segments i and j adjacent (share a vertex)?"""

    if i == j:
        return True
    if closed:
        return (j == (i + 1) % m) or (i == (j + 1) % m)
    else:
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
    Equations (15)-(16) in Klenin & Langowski (2000).
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
    Equations (17)-(25) in Klenin & Langowski (2000).
    """

    def F(t1: float, t2: float) -> float:
        rad = t1 * t1 + t2 * t2 - 2.0 * t1 * t2 * cosb + a0 * a0 * sin2b
        if rad <= 0.0:  # pragma: no cover
            return 0.0
        denom = a0 * math.sqrt(rad)
        num = t1 * t2 + a0 * a0 * cosb
        # Eq (25): -(1/4π) * arctan( num / denom )
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


def writhe_method_1a(
    points: npt.ArrayLike, *, closed: bool = True, eps: float = 1e-12
) -> tuple[float, float]:
    """
    Method 1a: pairwise solid angles (Eqs 13, 15-16).
    """

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
            V, Vabs = _gauss_pair_method_1a(A[i], B[i], A[j], B[j], eps=eps)
            sumV += V
            sumVabs += Vabs

    pref = 1.0 / (2.0 * math.pi)  # from Eq (13): Wr = sum_{i<j} V_ij /(2π)
    writhe = sumV * pref
    acn = sumVabs * pref

    writhe = _clean_writhe(writhe, acn, atol=eps)

    return writhe, acn


def writhe_method_1b(
    points: npt.ArrayLike, *, closed: bool = True, eps: float = 1e-12
) -> tuple[float, float]:
    """
    Method 1b: analytic Gauss integral (Eqs 13, 24-25).
    Returns (writhe, acn_like).
    """
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

    acn = sumVabs * pref
    writhe = _clean_writhe(sumV * pref, acn, atol=eps)

    return writhe, acn


def writhe_method_2a(points: npt.ArrayLike, *, eps: float = 1e-12) -> float:
    """
    Method 2a: Wr = Twz + Wrz - Tw (Eqs 30-34), z-axis projection. Requires a
    closed chain.
    """

    P = _as_points(points)
    n = len(P)
    if n < 4:
        return 0.0

    A, B = _segments(P, closed=True)
    S = B - A

    # p_i = unit normal to (s_{i-1}, s_i) (Eq 27)
    p = np.zeros((n, 3), dtype=float)
    for i in range(n):
        c = np.cross(S[i - 1], S[i])
        u = _unit(c, eps)
        if u is None:
            return float("nan")
        p[i] = u

    # Tw (Eq 30)
    Tw_sum = 0.0
    for i in range(n):
        pi = p[i]
        pip = p[(i + 1) % n]
        ang = math.acos(float(np.clip(np.dot(pi, pip), -1.0, 1.0)))
        sgn = float(np.sign(np.dot(pi, S[(i + 1) % n])))
        Tw_sum += ang * sgn
    Tw = Tw_sum / (2.0 * math.pi)

    # Twz (Eq 31)
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
    Requires CLOSED chain.
    """

    P = _as_points(points)
    n = len(P)
    if n < 4:
        return 0.0

    A, B = _segments(P, closed=True)
    S = B - A

    k = np.array([0.0, 0.0, 1.0], dtype=float)

    # a_i (Eq 35)
    a = np.zeros((n, 3), dtype=float)
    for i in range(n):
        u = _unit(np.cross(k, S[i]), eps)
        if u is None:
            return float("nan")
        a[i] = u

    # p_i (Eq 27)
    p = np.zeros((n, 3), dtype=float)
    for i in range(n):
        u = _unit(np.cross(S[i - 1], S[i]), eps)
        if u is None:
            return float("nan")
        p[i] = u

    # Tw (Eq 37)
    Tw_sum = 0.0
    for i in range(n):
        pi = p[i]
        term1 = math.acos(float(np.clip(np.dot(a[i - 1], pi), -1.0, 1.0)))
        term2 = math.acos(float(np.clip(np.dot(pi, a[i]), -1.0, 1.0)))
        Tw_sum += (term1 - term2) * float(np.sign(pi[2]))
    Tw = Tw_sum / (2.0 * math.pi)

    Wrz = _directional_writhe_Wrz(P, eps=eps)
    return float(Wrz - Tw)
