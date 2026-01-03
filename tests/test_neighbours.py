import numpy as np

from topo_metrics.neighbours import autoreduce_neighborlist


def test_autoreduce_neighborlist_remove_by_type():
    """Test removing atoms by type (e.g., oxygen atoms)."""

    # Simple structure: 3 atoms in a line: Si-O-Si
    # Atom 0 (Si), Atom 1 (O), Atom 2 (Si)
    frac_coords = np.array(
        [
            [0.0, 0.0, 0.0],  # Si
            [0.5, 0.0, 0.0],  # O
            [1.0, 0.0, 0.0],  # Si
        ]
    )
    cart_coords = [None, None, None]
    symbols = ["Si", "O", "Si"]

    # Edges (1-based): Si-O, O-Si with no shifts
    edges = np.array(
        [
            [1, 2, 0, 0, 0],  # Si(0) -> O(1)
            [2, 3, 0, 0, 0],  # O(1) -> Si(2)
        ],
        dtype=int,
    )

    # Remove oxygen atoms
    _, _, new_symbols, new_edges, old_to_new = autoreduce_neighborlist(
        cart_coords, frac_coords, symbols, edges, remove_types={"O"}
    )

    # Should have 2 Si atoms left
    assert len(new_symbols) == 2
    assert new_symbols == ["Si", "Si"]

    # Should have 1 edge connecting the two Si atoms
    assert new_edges.shape[0] == 1
    assert new_edges[0, 0] == 1  # First Si (1-based)
    assert new_edges[0, 1] == 2  # Second Si (1-based)

    # Check mapping
    assert old_to_new[0] == 0  # Si at index 0 -> new index 0
    assert old_to_new[1] == -1  # O removed
    assert old_to_new[2] == 1  # Si at index 2 -> new index 1


def test_autoreduce_neighborlist_degree2():
    """Test removing degree-2 atoms."""

    # Chain: A-B-C where B has degree 2
    frac_coords = np.array(
        [
            [0.0, 0.0, 0.0],  # A
            [0.5, 0.0, 0.0],  # B (degree 2)
            [1.0, 0.0, 0.0],  # C
        ]
    )
    cart_coords = [None, None, None]
    symbols = ["A", "B", "C"]

    edges = np.array(
        [
            [1, 2, 0, 0, 0],  # A -> B
            [2, 3, 0, 0, 0],  # B -> C
        ],
        dtype=int,
    )

    _, _, new_symbols, new_edges, _ = autoreduce_neighborlist(
        cart_coords,
        frac_coords,
        symbols,
        edges,
        remove_types={"O"},
        remove_degree2=True,
    )

    # B should be removed
    assert len(new_symbols) == 2
    assert "B" not in new_symbols

    # Should have direct edge A-C
    assert new_edges.shape[0] == 1


def test_autoreduce_neighborlist_periodic():
    """Test with periodic boundary conditions (non-zero shifts)."""

    frac_coords = np.array(
        [
            [0.0, 0.0, 0.0],  # Si
            [0.5, 0.0, 0.0],  # O
            [0.9, 0.0, 0.0],  # Si
        ]
    )
    cart_coords = [None, None, None]
    symbols = ["Si", "O", "Si"]

    # Edge with shift vector
    edges = np.array(
        [
            [1, 2, 0, 0, 0],  # Si -> O
            [2, 3, 1, 0, 0],  # O -> Si (with shift)
        ],
        dtype=int,
    )

    _, _, _, new_edges, _ = autoreduce_neighborlist(
        cart_coords, frac_coords, symbols, edges, remove_types={"O"}
    )

    # Check shift is preserved
    assert new_edges.shape[0] == 1
    assert new_edges[0, 2] == 1  # Shift in x
    assert new_edges[0, 3] == 0  # No shift in y
    assert new_edges[0, 4] == 0  # No shift in z


def test_autoreduce_neighborlist_empty_removals():
    """Test when nothing is removed."""

    frac_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
        ]
    )
    cart_coords = [None, None]
    symbols = ["A", "B"]
    edges = np.array([[1, 2, 0, 0, 0]], dtype=int)

    _, _, new_symbols, new_edges, old_to_new = autoreduce_neighborlist(
        cart_coords,
        frac_coords,
        symbols,
        edges,
        remove_types={"C"},  # Not present
    )

    # Everything should remain
    assert len(new_symbols) == 2
    assert new_symbols == ["A", "B"]
    assert new_edges.shape[0] == 1
    np.testing.assert_array_equal(old_to_new, [0, 1])


def test_autoreduce_neighborlist_triangle():
    """Test with a triangular structure (clique formation)."""

    # Three atoms: A, B (remove), C
    # B connects to both A and C, so removing B should create A-C edge
    frac_coords = np.array(
        [
            [0.0, 0.0, 0.0],  # A
            [0.5, 0.5, 0.0],  # B (to remove)
            [1.0, 0.0, 0.0],  # C
        ]
    )
    cart_coords = [None, None, None]
    symbols = ["A", "B", "C"]

    edges = np.array(
        [
            [1, 2, 0, 0, 0],  # A -> B
            [2, 3, 0, 0, 0],  # B -> C
        ],
        dtype=int,
    )

    _, _, new_symbols, new_edges, _ = autoreduce_neighborlist(
        cart_coords, frac_coords, symbols, edges, remove_types={"B"}
    )

    # A and C remain, with direct edge
    assert len(new_symbols) == 2
    assert new_symbols == ["A", "C"]
    assert new_edges.shape[0] == 1


def test_autoreduce_neighborlist_no_edges():
    """Test with isolated atoms (no edges)."""

    frac_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
        ]
    )
    cart_coords = [None, None]
    symbols = ["A", "B"]
    edges = np.array([], dtype=int).reshape(0, 5)

    _, _, new_symbols, new_edges, _ = autoreduce_neighborlist(
        cart_coords, frac_coords, symbols, edges, remove_types={"A"}
    )

    # Isolated atoms (with no edges) are not removed by type
    # Both A and B remain since neither has edges
    assert len(new_symbols) == 2
    assert new_symbols == ["A", "B"]
    assert new_edges.shape[0] == 0


def test_autoreduce_neighborlist_complex_shifts():
    """Test complex shift combinations."""

    # Four atoms in a chain with various shifts
    frac_coords = np.array(
        [
            [0.0, 0.0, 0.0],  # A
            [0.5, 0.0, 0.0],  # B (remove)
            [1.0, 0.0, 0.0],  # C (remove)
            [1.5, 0.0, 0.0],  # D
        ]
    )
    cart_coords = [None] * 4
    symbols = ["A", "B", "C", "D"]

    edges = np.array(
        [
            [1, 2, 0, 0, 0],  # A -> B
            [2, 3, 1, 0, 0],  # B -> C (shift)
            [3, 4, 0, 1, 0],  # C -> D (different shift)
        ],
        dtype=int,
    )

    _, _, new_symbols, new_edges, _ = autoreduce_neighborlist(
        cart_coords, frac_coords, symbols, edges, remove_types={"B", "C"}
    )

    # Only A and D remain
    assert len(new_symbols) == 2
    assert new_symbols == ["A", "D"]

    # Should have combined shift
    assert new_edges.shape[0] == 1
    assert new_edges[0, 2] == 1  # x shift
    assert new_edges[0, 3] == 1  # y shift


def test_autoreduce_neighborlist_self_loop_prevention():
    """Test that self-loops are prevented when contracting."""

    # Two atoms both of type B connected to each other
    frac_coords = np.array(
        [
            [0.0, 0.0, 0.0],  # B
            [0.5, 0.5, 0.5],  # B
        ]
    )
    cart_coords = [None] * 4
    symbols = ["B", "B"]

    edges = np.array(
        [
            [1, 2, 0, 0, 0],  # B -> B
        ],
        dtype=int,
    )

    _, _, new_symbols, new_edges, _ = autoreduce_neighborlist(
        cart_coords, frac_coords, symbols, edges, remove_types={"B"}
    )

    # All atoms removed
    assert len(new_symbols) == 0
    assert new_edges.shape[0] == 0


def test_autoreduce_neighborlist_degree2_chain():
    """Test removal of multiple degree-2 atoms in sequence."""

    # Chain: A-B-C-D-E where B, C, D all have degree 2
    frac_coords = np.array(
        [
            [0.0, 0.0, 0.0],  # A
            [0.2, 0.0, 0.0],  # B
            [0.4, 0.0, 0.0],  # C
            [0.6, 0.0, 0.0],  # D
            [0.8, 0.0, 0.0],  # E
        ]
    )
    cart_coords = [None] * 5
    symbols = ["A", "B", "C", "D", "E"]

    edges = np.array(
        [
            [1, 2, 0, 0, 0],  # A -> B
            [2, 3, 0, 0, 0],  # B -> C
            [3, 4, 0, 0, 0],  # C -> D
            [4, 5, 0, 0, 0],  # D -> E
        ],
        dtype=int,
    )

    _, _, new_symbols, new_edges, _ = autoreduce_neighborlist(
        cart_coords, frac_coords, symbols, edges, remove_degree2=True
    )

    # Only A and E should remain
    assert len(new_symbols) == 2
    assert new_symbols == ["A", "E"]

    # Should have direct edge A-E
    assert new_edges.shape[0] == 1


def test_autoreduce_neighborlist_combined_removal():
    """Test combining type-based and degree-2 removal."""

    # A-O-C-D where O is type O (to remove) and C has degree 2
    frac_coords = np.array(
        [
            [0.0, 0.0, 0.0],  # A
            [0.3, 0.0, 0.0],  # O
            [0.6, 0.0, 0.0],  # C
            [0.9, 0.0, 0.0],  # D
        ]
    )
    cart_coords = [None] * 4
    symbols = ["A", "O", "C", "D"]

    edges = np.array(
        [
            [1, 2, 0, 0, 0],  # A -> O
            [2, 3, 0, 0, 0],  # O -> C
            [3, 4, 0, 0, 0],  # C -> D
        ],
        dtype=int,
    )

    _, _, new_symbols, _, _ = autoreduce_neighborlist(
        cart_coords,
        frac_coords,
        symbols,
        edges,
        remove_types={"O"},
        remove_degree2=True,
    )

    # After O is removed: A-C-D with C having degree 2
    # Then C is removed (degree-2 removal): A-D remain
    assert len(new_symbols) == 2
    assert "O" not in new_symbols
    assert "C" not in new_symbols
    assert set(new_symbols) == {"A", "D"}
