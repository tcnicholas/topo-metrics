import numpy as np
import pytest

import topo_metrics.knots as k
from topo_metrics.knots import (
    writhe_method_1a,
    writhe_method_1b,
    writhe_method_2a,
    writhe_method_2b,
)


def square_xy():
    # Planar unknot (square) in xy-plane
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )


def ring_example_points():
    # 12-point example (unique points, no repeated endpoint)
    return np.array(
        [
            [2.24125, 4.4825, 0.0],
            [4.4825, 6.72375, 0.0],
            [4.4825, 8.965, -2.24125],
            [6.72375, 8.965, -4.4825],
            [4.4825, 8.965, -6.72375],
            [4.4825, 6.72375, -8.965],
            [6.72375, 4.4825, -8.965],
            [4.4825, 2.24125, -8.965],
            [4.4825, 0.0, -6.72375],
            [2.24125, 0.0, -4.4825],
            [4.4825, 0.0, -2.24125],
            [4.4825, 2.24125, 0.0],
        ],
        dtype=float,
    )


def torus_wave(N=80, R=3.0, r=1.0, k=3):
    """Smooth non-planar closed curve (unknot on a torus-like tube)."""

    t = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    x = (R + r * np.cos(k * t)) * np.cos(t)
    y = (R + r * np.cos(k * t)) * np.sin(t)
    z = r * np.sin(k * t)
    return np.column_stack([x, y, z]).astype(float)


def test_segments_open_len_lt_2():
    # len(points) == 0
    P0 = np.zeros((0, 3), dtype=float)
    A, B = k._segments(P0, closed=False)
    assert A.shape == (0, 3)
    assert B.shape == (0, 3)

    # len(points) == 1
    P1 = np.array([[0.0, 0.0, 0.0]], dtype=float)
    A, B = k._segments(P1, closed=False)
    assert A.shape == (0, 3)
    assert B.shape == (0, 3)


def test_are_adjacent_i_equals_j():
    assert k._are_adjacent(2, 2, m=10, closed=True) is True
    assert k._are_adjacent(0, 0, m=10, closed=False) is True


def test_planar_square_closed_writhe_zero_all_methods():
    P = square_xy()

    wr1a, acn1a = writhe_method_1a(P, closed=True)
    wr1b, acn1b = writhe_method_1b(P, closed=True)
    wr2a = writhe_method_2a(P)
    wr2b = writhe_method_2b(P)

    # Planar closed curve => writhe ~ 0
    assert wr1a == pytest.approx(0.0, abs=1e-12)
    assert wr1b == pytest.approx(0.0, abs=1e-12)
    assert wr2a == pytest.approx(0.0, abs=1e-12)
    assert wr2b == pytest.approx(0.0, abs=1e-12)

    # Planar embedded curve => ACN ~ 0
    assert acn1a == pytest.approx(0.0, abs=1e-12)
    assert acn1b == pytest.approx(0.0, abs=1e-12)

    # 1a and 1b should agree
    assert wr1a == pytest.approx(wr1b, abs=1e-10)
    assert acn1a == pytest.approx(acn1b, abs=1e-10)


def test_open_vs_closed_matches_known_example():
    P = ring_example_points()

    wr_open, acn_open = writhe_method_1a(P, closed=False)
    wr_closed, acn_closed = writhe_method_1a(P, closed=True)

    assert wr_open == pytest.approx(-0.13841097891704496, abs=1e-12)
    assert wr_closed == pytest.approx(0.0, abs=1e-12)

    assert acn_open > 0.0
    assert acn_closed > 0.0


def test_methods_consistent_on_generic_nonplanar_curve():
    P = torus_wave(N=80)

    wr1a, acn1a = writhe_method_1a(P, closed=True)
    wr1b, acn1b = writhe_method_1b(P, closed=True)

    # 2a/2b require the curve not to hit their axis singularities.
    wr2a = writhe_method_2a(P)
    wr2b = writhe_method_2b(P)

    assert not np.isnan(wr2a)
    assert not np.isnan(wr2b)

    # 1a and 1b should match tightly
    assert wr1a == pytest.approx(wr1b, abs=1e-8)
    assert acn1a == pytest.approx(acn1b, abs=1e-8)

    # 2a and 2b should match 1a reasonably well
    assert wr2a == pytest.approx(wr1a, abs=1e-6)
    assert wr2b == pytest.approx(wr1a, abs=1e-6)


def test_reverse_curve_keeps_writhe_and_acn():
    P = torus_wave(N=80)

    wr, acn = writhe_method_1a(P, closed=True)
    wr_r, acn_r = writhe_method_1a(P[::-1], closed=True)

    assert wr_r == pytest.approx(wr, abs=1e-10)
    assert acn_r == pytest.approx(acn, abs=1e-10)


def test_translation_and_scaling_invariance():
    P = torus_wave(N=80)

    wr, acn = writhe_method_1a(P, closed=True)

    P2 = P + np.array([10.0, -3.0, 5.5])  # translation
    wr_t, acn_t = writhe_method_1a(P2, closed=True)

    P3 = 7.25 * P  # scaling
    wr_s, acn_s = writhe_method_1a(P3, closed=True)

    assert wr_t == pytest.approx(wr, abs=1e-10)
    assert acn_t == pytest.approx(acn, abs=1e-10)

    assert wr_s == pytest.approx(wr, abs=1e-10)
    assert acn_s == pytest.approx(acn, abs=1e-10)


def test_duplicate_endpoint_is_ignored():
    P = torus_wave(N=80)
    Pdup = np.vstack([P, P[0]])  # explicit closure

    wr, acn = writhe_method_1a(P, closed=True)
    wr2, acn2 = writhe_method_1a(Pdup, closed=True)

    assert wr2 == pytest.approx(wr, abs=1e-10)
    assert acn2 == pytest.approx(acn, abs=1e-10)


def test_method2b_nan_when_segment_parallel_to_z_axis():
    # Build a closed loop with some segments parallel to z:
    # This triggers k × s_i = 0 for those vertical segments in method 2b.
    P = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],  # vertical segment
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    wr2b = writhe_method_2b(P)
    assert np.isnan(wr2b)


def test_invalid_shape_raises():
    with pytest.raises(ValueError):
        writhe_method_1a(np.zeros((3, 2)), closed=True)


def test_directional_writhe_triple_positive_and_negative():
    # monkeypatch _are_adjacent so code doesn't skip chosen (i,j),
    # and monkeypatch _cross2 so denom != 0 and ti,tj fall in (0,1).
    #
    # Then choose A/S so that:
    # triple = dot(cross(S[j], S[i]), (A[j]-A[i])) is >0 in one test and <0 in
    # another test.

    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    def fake_segments(_points, closed=True):
        A = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
        B = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
        return A, B

    def never_adjacent(i, j, m, closed):
        return False

    # Make denom=1.0 and choose ti=tj=0.5 by setting cross2 results
    # denom = cross2(r,s)
    # ti = cross2(qp,s)/denom
    # tj = cross2(qp,r)/denom
    # We'll return 1.0 for denom, 0.5 for the other two.
    calls = {"n": 0}

    def cross2_stub(a, b):
        calls["n"] += 1
        if calls["n"] == 1:
            return 1.0
        return 0.5

    # POSITIVE triple: choose S0=(1,0,0), S1=(0,1,0), A1-A0=(0,0,1)
    # cross(S1,S0) = cross((0,1,0),(1,0,0)) = (0,0,-1)
    # dot((0,0,-1),(0,0,1)) = -1  (that's negative)
    # So to make it positive, swap S0/S1:
    # cross(S1,S0) with S1=(1,0,0), S0=(0,1,0) => (0,0,1), dot(...,(0,0,1))=+1
    def segments_positive(_points, closed=True):
        A = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        S = np.array(
            [
                [0.0, 1.0, 0.0],  # S[i]
                [1.0, 0.0, 0.0],
            ],  # S[j]
            dtype=float,
        )
        B = A + S
        return A, B

    # NEGATIVE triple: flip one segment so cross changes sign
    def segments_negative(_points, closed=True):
        A = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        S = np.array(
            [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],  # flipped
            dtype=float,
        )
        B = A + S
        return A, B

    # --- test triple > 0 increments ---
    calls["n"] = 0
    import topo_metrics.knots as km

    orig_segments = km._segments
    orig_adj = km._are_adjacent
    orig_cross2 = km._cross2
    try:
        km._segments = segments_positive
        km._are_adjacent = never_adjacent
        km._cross2 = cross2_stub
        wrz = km._directional_writhe_Wrz(points)
        assert wrz == 1.0
    finally:
        km._segments = orig_segments
        km._are_adjacent = orig_adj
        km._cross2 = orig_cross2

    # --- test triple < 0 decrements ---
    calls["n"] = 0
    try:
        km._segments = segments_negative
        km._are_adjacent = never_adjacent
        km._cross2 = cross2_stub
        wrz = km._directional_writhe_Wrz(points)
        assert wrz == -1.0
    finally:
        km._segments = orig_segments
        km._are_adjacent = orig_adj
        km._cross2 = orig_cross2


def test_unit_returns_none_for_zero_vector():
    v = np.array([0.0, 0.0, 0.0], dtype=float)
    assert k._unit(v, eps=1e-12) is None


def test_clean_writhe_thresholding():
    # small wr cleaned to 0
    assert k._clean_writhe(1e-20, acn=1.0, atol=1e-12) == 0.0
    # large wr not cleaned
    assert k._clean_writhe(1e-6, acn=0.0, atol=1e-12) == pytest.approx(1e-6)


def test_gauss_pair_method_1a_degenerate_normals_returns_zero():
    # Collinear points -> cross products vanish -> n? == None -> returns (0,0)
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.0, 0.0, 0.0])
    p3 = np.array([2.0, 0.0, 0.0])
    p4 = np.array([3.0, 0.0, 0.0])

    V, Vabs = k._gauss_pair_method_1a(p1, p2, p3, p4)
    assert V == 0.0
    assert Vabs == 0.0


def test_gauss_pair_method_1a_triple_zero_signed_zero_but_abs_positive():
    # Constructed so scalar triple product == 0 but normals are defined.
    # This covers the `sgn = ... else 0.0` path in 1a.
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([0.0, 1.0, 1.0])
    p3 = np.array([0.0, 0.0, 1.0])
    p4 = np.array([0.0, 1.0, 0.0])

    V, Vabs = k._gauss_pair_method_1a(p1, p2, p3, p4)
    assert V == 0.0
    assert Vabs > 0.0


def test_gauss_pair_method_1b_zero_length_segment_returns_zero():
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([0.0, 0.0, 0.0])  # zero-length
    p3 = np.array([0.0, 1.0, 0.0])
    p4 = np.array([1.0, 1.0, 0.0])

    V, Vabs = k._gauss_pair_method_1b(p1, p2, p3, p4)
    assert V == 0.0
    assert Vabs == 0.0


def test_gauss_pair_method_1b_parallel_segments_returns_zero():
    # s1 and s2 parallel => sin^2(beta) ~ 0 -> returns (0,0)
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.0, 0.0, 0.0])
    p3 = np.array([0.0, 1.0, 0.0])
    p4 = np.array([1.0, 1.0, 0.0])

    V, Vabs = k._gauss_pair_method_1b(p1, p2, p3, p4)
    assert V == 0.0
    assert Vabs == 0.0


def test_gauss_pair_method_1b_a0_zero_returns_zero():
    # Make e1 and e2 non-parallel, but r12 lies in their span => a0 = 0 =>
    # returns (0,0).
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.0, 0.0, 0.0])  # e1 along x
    p3 = np.array([0.0, 0.0, 0.0])  # r12 = 0 => a0 = 0
    p4 = np.array([0.0, 1.0, 0.0])  # e2 along y (non-parallel)

    V, Vabs = k._gauss_pair_method_1b(p1, p2, p3, p4)
    assert V == 0.0
    assert Vabs == 0.0


def test_method2a_nan_when_consecutive_segments_collinear():
    # S[0] and S[1] collinear => cross(S[i-1],S[i]) can be zero => _unit(None)
    # => nan
    P = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],  # collinear step
            [2.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    assert np.isnan(k.writhe_method_2a(P))


def test_method2b_nan_when_consecutive_segments_collinear_in_p_i():
    # Same idea, but exercises the p_i computation in method 2b.
    P = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],  # collinear step
            [2.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    assert np.isnan(k.writhe_method_2b(P))


def test_all_methods_two_points_return_zero():
    P = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
        ],
        dtype=float,
    )

    wr1a, acn1a = writhe_method_1a(P, closed=False)
    wr1b, acn1b = writhe_method_1b(P, closed=False)
    wr2a = writhe_method_2a(P)
    wr2b = writhe_method_2b(P)

    assert wr1a == 0.0 and acn1a == 0.0
    assert wr1b == 0.0 and acn1b == 0.0
    assert wr2a == 0.0
    assert wr2b == 0.0


def test_directional_writhe_triple_zero_hits_elif_false_arc():
    # Planar bow-tie: segments 0 and 2 cross in xy, but triple == 0
    P = np.array(
        [
            [0.0, 0.0, 0.0],  # P0
            [1.0, 1.0, 0.0],  # P1   seg0: P0->P1 (diag)
            [0.0, 1.0, 0.0],  # P2
            [1.0, 0.0, 0.0],  # P3   seg2: P2->P3 (other diag)
        ],
        dtype=float,
    )

    # triple == 0 => neither +1 nor -1, so Wrz stays 0,
    # but the `elif triple < 0` condition is still evaluated (false) and loops.
    assert k._directional_writhe_Wrz(P) == pytest.approx(0.0, abs=1e-12)
