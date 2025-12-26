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


def test_vertex_symbol_single_ring_size():
    """Test VertexSymbol with a single ring size (no parentheses)."""

    vs = VertexSymbol(
        vector=[[6, 6, 6], [4]], vector_all_rings=[[6, 6, 6], [4]]
    )

    # Single ring size should not have parentheses
    assert vs.to_str() == "[6(3).4]"


def test_vertex_symbol_repr():
    """Test VertexSymbol __repr__ method."""

    vs = VertexSymbol(vector=[[6, 6], [4]], vector_all_rings=[[6, 6, 8], [4]])

    repr_str = repr(vs)
    assert "VertexSymbol" in repr_str
    assert "VS" in repr_str
    assert "VS(all_rings)" in repr_str


def test_carvs_repr():
    """Test CARVS __repr__ method."""

    c = CARVS(vector=np.array([1, 2]), spread=0.5, is_single_node=True)
    repr_str = repr(c)

    assert "CARVS" in repr_str
    assert "{" in repr_str


def test_carvs_str_integer_count():
    """Test CARVS string representation with integer counts."""

    c = CARVS(
        vector=np.array([1.0, 2.0, 3.0]), spread=0.0, is_single_node=False
    )
    s = str(c)

    # Should format counts as integers when they're whole numbers
    assert "{1.2(2).3(3)}" in s


def test_carvs_str_fractional_count():
    """Test CARVS string representation with fractional counts."""

    c = CARVS(vector=np.array([1.5, 2.3]), spread=0.0, is_single_node=False)
    s = str(c)

    # Should format fractional counts with one decimal place in the exact
    # expected format
    expected_prefix = "{1(1.5).2(2.3)}"
    assert s == expected_prefix or s.startswith(expected_prefix + " σ=")


def test_carvs_from_list_different_lengths():
    """Test CARVS.from_list with vectors of different lengths."""

    c1 = CARVS(vector=np.array([1, 2]), spread=0.5, is_single_node=True)
    c2 = CARVS(vector=np.array([3, 4, 5]), spread=1.0, is_single_node=True)
    result = CARVS.from_list([c1, c2])

    # Result should have length of longest vector (3)
    assert len(result.vector) == 3
    # Check averaging
    np.testing.assert_allclose(result.vector, [2.0, 3.0, 2.5])


def test_carvs_from_list_single_item():
    """Test CARVS.from_list with a single CARVS object."""

    c = CARVS(vector=np.array([1, 2, 3]), spread=0.5, is_single_node=True)
    result = CARVS.from_list([c])

    np.testing.assert_allclose(result.vector, c.vector)
    assert result.spread == c.spread
    assert result.is_single_node == c.is_single_node


def test_pad_carvs_already_same_length():
    """Test pad_carvs when all vectors already have same length."""

    c1 = CARVS(vector=np.array([1, 2]), spread=0.0, is_single_node=True)
    c2 = CARVS(vector=np.array([3, 4]), spread=0.0, is_single_node=True)
    result = pad_carvs([c1, c2])

    assert len(result[0].vector) == 2
    assert len(result[1].vector) == 2


def test_pad_carvs_per_atom_single_array():
    """Test pad_carvs_per_atom with a single array."""

    a = [np.ones((2, 3))]
    result = pad_carvs_per_atom(a)

    assert result[0].shape == (2, 3)


def test_pad_carvs_per_atom_different_heights():
    """Test pad_carvs_per_atom with arrays of different heights."""

    a = [np.ones((2, 2)), np.ones((3, 2))]
    result = pad_carvs_per_atom(a)

    # Width should match
    assert all(r.shape[1] == 2 for r in result)


def test_topological_distances_single_carvs():
    """Test get_all_topological_distances with a single CARVS."""

    c = CARVS(vector=np.array([1, 2]), spread=0, is_single_node=False)
    dists = get_all_topological_distances([c])

    assert dists.shape == (1, 1)
    assert np.isclose(dists[0, 0], 0)


def test_topological_distances_identical_carvs():
    """Test get_all_topological_distances with identical CARVS objects."""

    c = CARVS(vector=np.array([1, 2]), spread=0, is_single_node=False)
    dists = get_all_topological_distances([c, c, c])

    # All distances should be zero
    np.testing.assert_allclose(dists, 0, atol=1e-10)


def test_topological_distances_symmetry():
    """Test that distance matrix is symmetric."""

    c1 = CARVS(vector=np.array([1, 0, 0]), spread=0, is_single_node=False)
    c2 = CARVS(vector=np.array([0, 1, 0]), spread=0, is_single_node=False)
    c3 = CARVS(vector=np.array([0, 0, 1]), spread=0, is_single_node=False)
    dists = get_all_topological_distances([c1, c2, c3])

    # Check symmetry
    np.testing.assert_allclose(dists, dists.T)


def test_vertex_symbol_empty_vector():
    """Test VertexSymbol with empty ring vectors."""

    vs = VertexSymbol(vector=[[], []], vector_all_rings=[[], []])

    # Should handle empty vectors gracefully
    result = vs.to_str()
    assert result == "[.]"


def test_vertex_symbol_multiple_multiplicities():
    """Test VertexSymbol with multiple ring sizes having multiplicities."""

    vs = VertexSymbol(
        vector=[[4, 4, 4, 6, 6]], vector_all_rings=[[4, 4, 4, 6, 6]]
    )

    result = vs.to_str()
    assert "4(3)" in result
    assert "6(2)" in result


def test_carvs_str_skip_zero_counts():
    """Test that CARVS string skips zero counts."""

    c = CARVS(vector=np.array([0, 1, 0, 2]), spread=0.0, is_single_node=False)
    s = str(c)

    # Should not include sizes with count < 1
    assert "1." not in s  # Size 1 has count 0 and must be excluded
    assert "3." not in s  # Size 3 has count 0


def test_carvs_str_spread_formatting():
    """Test CARVS string formatting of spread value."""

    c = CARVS(vector=np.array([1, 2]), spread=1.23456, is_single_node=False)
    s = str(c)

    # Spread should be formatted to 1 decimal place
    assert "σ=1.2" in s
