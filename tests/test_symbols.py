import numpy as np
import pytest

from topo_metrics.symbols import (
    CARVS,
    VertexSymbol,
    get_all_topological_distances,
    pad_carvs,
    pad_carvs_per_atom,
)


def test_vertex_symbol_to_str_basic():
    """Test basic string representation of VertexSymbol."""

    vs = VertexSymbol(
        vector=[[6, 6, 8], [4, 4]], vector_all_rings=[[4, 6, 6, 8], [4, 4, 6]]
    )

    assert vs.to_str() == "[(6(2),8).4(2)]"
    assert vs.to_str(all_rings=True) == "[(4,6(2),8).(4(2),6)]"


def test_carvs_from_list_raises_on_empty():
    """Test that CARVS.from_list raises an error when given an empty list."""

    with pytest.raises(
        ValueError, match="Cannot create a CARVS from an empty list."
    ):
        CARVS.from_list([])


def test_carvs_from_list_merging():
    """Test merging of multiple CARVS objects into a single object."""

    c1 = CARVS(vector=np.array([1, 2]), spread=0.5, is_single_node=True)
    c2 = CARVS(vector=np.array([3, 4]), spread=1.0, is_single_node=False)
    result = CARVS.from_list([c1, c2])

    np.testing.assert_allclose(result.vector, [2, 3])
    assert np.isclose(result.spread, 0.75)
    assert result.is_single_node is False


def test_carvs_str_formats_correctly():
    """Test that the string representation of CARVS is formatted correctly."""

    c = CARVS(
        vector=np.array([1, 2.0, 3.5, 0.0]), spread=0.1, is_single_node=True
    )
    s = str(c)

    assert s.startswith("{1.2(2).3(3.5)}")
    assert "σ=" in s


def test_carvs_str_excludes_zero_spread():
    """Test that the string representation of CARVS excludes zero spread."""

    carvs = CARVS(vector=np.array([1, 2]), spread=0.0, is_single_node=False)
    result = str(carvs)

    assert "σ" not in result


def test_pad_carvs_lengths():
    """Test that pad_carvs pads vectors to the same length."""

    c1 = CARVS(vector=np.array([1, 2]), spread=0.0, is_single_node=True)
    c2 = CARVS(vector=np.array([1]), spread=0.0, is_single_node=True)
    result = pad_carvs([c1, c2])

    assert all(len(c.vector) == 2 for c in result)


def test_pad_carvs_per_atom():
    """Test that pad_carvs_per_atom pads vectors to the same length."""

    a = [np.ones((2, 2)), np.ones((2, 1))]
    result = pad_carvs_per_atom(a)

    assert all(r.shape[1] == 2 for r in result)


def test_topological_distances_matrix():
    """Test get_all_topological_distances generates correct distance matrix."""

    c1 = CARVS(vector=np.array([1, 0]), spread=0, is_single_node=False)
    c2 = CARVS(vector=np.array([0, 1]), spread=0, is_single_node=False)
    dists = get_all_topological_distances([c1, c2])

    assert dists.shape == (2, 2)
    assert np.isclose(dists[0, 1], dists[1, 0])
    assert np.isclose(dists[0, 0], 0)


def test_topological_distances_type_check():
    """Test get_all_topological_distances raises an error on invalid input."""

    with pytest.raises(TypeError):
        get_all_topological_distances("not a list")

    with pytest.raises(TypeError):
        get_all_topological_distances(
            [CARVS(vector=np.array([1]), spread=0, is_single_node=True), 123]
        )
