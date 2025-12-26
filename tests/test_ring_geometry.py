import numpy as np
import pytest

from topo_metrics.ring_geometry import RingGeometry
from topo_metrics.topology import Node


def make_square_ring_nodes():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    nodes = tuple(
        Node(node_id=i, cart_coord=coords[i]) for i in range(len(coords))
    )
    return nodes, coords


def test_species_and_len():

    nodes, _ = make_square_ring_nodes()
    rg = RingGeometry(nodes=nodes)

    assert len(rg) == 4
    assert rg.species == "SiSiSiSi"


def test_radius_of_gyration_square():
    nodes, _ = make_square_ring_nodes()
    rg = RingGeometry(nodes=nodes)

    # unit square corners around centroid: Rg^2 = 0.5 -> Rg = sqrt(0.5)
    assert rg.radius_of_gyration==pytest.approx(np.sqrt(0.5), rel=0, abs=1e-12)


def test_gyration_tensor_square():
    nodes, _ = make_square_ring_nodes()
    rg = RingGeometry(nodes=nodes)

    Q = rg.gyration_tensor
    expected = np.array(
        [
            [0.25, 0.0, 0.0],
            [0.0, 0.25, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    assert np.allclose(Q, expected, atol=1e-12, rtol=0.0)


def test_principal_moments_square():
    nodes, _ = make_square_ring_nodes()
    rg = RingGeometry(nodes=nodes)
    evals = rg.principal_moments
    assert np.allclose(evals, np.array([0.0, 0.25, 0.25]), atol=1e-12, rtol=0.0)


def test_asphericity_square():
    nodes, _ = make_square_ring_nodes()
    rg = RingGeometry(nodes=nodes)

    # For λ = (0,0.25,0.25) this definition yields 0.25
    assert rg.asphericity == pytest.approx(0.25, rel=0, abs=1e-12)


def test_geometric_centroid_square():
    nodes, _ = make_square_ring_nodes()
    rg = RingGeometry(nodes=nodes)

    assert np.allclose(
        rg.geometric_centroid, np.array([0.5, 0.5, 0.0]), atol=1e-12, rtol=0.0
    )


def test_writhe_and_acn_dispatch(monkeypatch):

    import topo_metrics.ring_geometry as rgmod

    nodes, coords = make_square_ring_nodes()
    rg = RingGeometry(nodes=nodes)

    called = {"1a": 0, "1b": 0, "2a": 0, "2b": 0}

    def f1a(P, closed=True):
        called["1a"] += 1
        assert np.allclose(P, coords)
        assert closed is False
        return (1.0, 2.0)

    def f1b(P, closed=True):
        called["1b"] += 1
        assert np.allclose(P, coords)
        assert closed is True
        return (3.0, 4.0)

    def f2a(P):
        called["2a"] += 1
        assert np.allclose(P, coords)
        return 5.0

    def f2b(P):
        called["2b"] += 1
        assert np.allclose(P, coords)
        return 6.0

    monkeypatch.setattr(rgmod, "writhe_method_1a", f1a)
    monkeypatch.setattr(rgmod, "writhe_method_1b", f1b)
    monkeypatch.setattr(rgmod, "writhe_method_2a", f2a)
    monkeypatch.setattr(rgmod, "writhe_method_2b", f2b)

    assert rg.writhe_and_acn(method="1a", closed=False) == (1.0, 2.0)
    assert rg.writhe_and_acn(method="1b", closed=True) == (3.0, 4.0)
    assert rg.writhe_and_acn(method="2a") == 5.0
    assert rg.writhe_and_acn(method="2b") == 6.0

    assert called == {"1a": 1, "1b": 1, "2a": 1, "2b": 1}


def test_writhe_and_acn_invalid_method_raises():
    nodes, _ = make_square_ring_nodes()
    rg = RingGeometry(nodes=nodes)

    with pytest.raises(ValueError):
        rg.writhe_and_acn(method="nope")


def test_to_xyz_writes_file(tmp_path):
    nodes, coords = make_square_ring_nodes()
    rg = RingGeometry(nodes=nodes)

    out = tmp_path / "ring.xyz"
    rg.to_xyz(out)

    assert out.exists()
    text = out.read_text()
    assert text.splitlines()[0].strip() == "4"
    assert any("0.0" in line or "1.0" in line for line in text.splitlines())


def test_asphericity_nan_when_s1_zero():
    # same position => gyration tensor is zero => eigenvalues = 0 => s1=0
    coord = np.array([1.23, -4.56, 7.89], dtype=float)
    nodes = tuple(Node(node_id=i, cart_coord=coord.copy()) for i in range(4))

    rg = RingGeometry(nodes=nodes)

    assert np.allclose(rg.gyration_tensor, np.zeros((3, 3)))
    assert np.allclose(rg.principal_moments, np.zeros(3))
    assert np.isnan(rg.asphericity)
