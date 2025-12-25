from typing import Final

import numpy as np
import numpy.typing as npt
import pytest
from pymatgen.core import Lattice as PymatgenLattice

from src.topo_metrics.io.cgd import parse_cgd, process_neighbour_list

TRUTH_LATTICE: Final = PymatgenLattice.from_parameters(
    a=4.9709, b=4.9709, c=6.9278, alpha=90, beta=90, gamma=90
)

TRUTH_ATOM_LABELS: Final[list[str]] = ["Si1", "Si2", "Si3", "Si4"]

TRUTH_COORDS: Final[list[list[float]]] = [
    [0.3005, 0.3005, 0.0],
    [0.1995, 0.8005, 0.25],
    [0.6995, 0.6995, 0.5],
    [0.8005, 0.1995, 0.75],
]

TRUTH_EDGES: Final[list[list[float | str]]] = [
    ["Si1", 0.8005, 0.1995, -0.25],
    ["Si1", -0.1995, 0.1995, -0.25],
    ["Si1", 0.1995, -0.1995, 0.25],
    ["Si1", 0.1995, 0.8005, 0.25],
    ["Si2", -0.3005, 0.6995, 0.5],
    ["Si2", 0.3005, 1.3005, 0.0],
    ["Si2", 0.6995, 0.6995, 0.5],
    ["Si2", 0.3005, 0.3005, 0.0],
    ["Si3", 0.8005, 0.1995, 0.75],
    ["Si3", 1.1995, 0.8005, 0.25],
    ["Si3", 0.8005, 1.1995, 0.75],
    ["Si3", 0.1995, 0.8005, 0.25],
    ["Si4", 0.6995, -0.3005, 0.5],
    ["Si4", 0.6995, 0.6995, 0.5],
    ["Si4", 1.3005, 0.3005, 1.0],
    ["Si4", 0.3005, 0.3005, 1.0],
]

TRUTH_NEIGHBOUR_LIST: Final[npt.NDArray[np.int_]] = np.array(
    [
        [1, 4, 0, 0, -1],
        [1, 4, -1, 0, -1],
        [1, 2, 0, -1, 0],
        [1, 2, 0, 0, 0],
        [2, 3, -1, 0, 0],
        [2, 1, 0, 1, 0],
        [2, 3, 0, 0, 0],
        [2, 1, 0, 0, 0],
        [3, 4, 0, 0, 0],
        [3, 2, 1, 0, 0],
        [3, 4, 0, 1, 0],
        [3, 2, 0, 0, 0],
        [4, 3, 0, -1, 0],
        [4, 3, 0, 0, 0],
        [4, 1, 1, 0, 1],
        [4, 1, 0, 0, 1],
    ]
)


def test_parse_cgd_file_not_found():
    """Test that an exception is raised when the CGD file is not found."""

    with pytest.raises(FileNotFoundError):
        parse_cgd("nonexistent_file.cgd")


def test_parse_cgd_with_blank_lines(tmp_path):
    test_file = tmp_path / "with_blank_lines.cgd"
    test_file.write_text(
        """
        cell 1 1 1 90 90 90

        atom A 1 0.0 0.0 0.0

        edge A 0.0 0.0 0.0

        """
    )
    lattice, atom_labels, coords, edges = parse_cgd(str(test_file))

    assert lattice is not None
    assert atom_labels == ["A"]
    assert coords.shape == (1, 3)
    assert len(edges) == 1


def test_parse_cgd(parsed_cgd):
    """Test the parsing of the CGD file."""

    lattice, atom_labels, all_coords, edges = parsed_cgd
    assert isinstance(lattice, PymatgenLattice)
    assert lattice == TRUTH_LATTICE
    assert atom_labels == TRUTH_ATOM_LABELS
    np.testing.assert_allclose(all_coords, TRUTH_COORDS)
    assert edges == TRUTH_EDGES


def test_neighbour_list(parsed_cgd):
    """Test the construction of the neighbour list from the parsed CGD file."""

    _, atom_labels, all_coords, edges = parsed_cgd
    neighbour_list = process_neighbour_list(edges, all_coords, atom_labels)
    np.testing.assert_equal(neighbour_list, TRUTH_NEIGHBOUR_LIST)


def test_process_neighbour_list_unknown_label():
    """Test that an exception is raised when an unknown label is encountered."""

    edges = [["X", 0.1, 0.1, 0.1]]  # Unknown label
    coords = np.array([[0.1, 0.1, 0.1]])
    with pytest.raises(ValueError, match="Unrecognised atom label"):
        process_neighbour_list(edges, coords, ["A"])


def test_process_neighbour_list_too_far():
    """Test an exception is raised when no atom is found within the cutoff."""

    edges = [["A", 0.9, 0.9, 0.9]]
    coords = np.array([[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="Could not find closest atom"):
        process_neighbour_list(edges, coords, ["A"])


def test_process_neighbour_list_without_coords():
    """Test an empty neighbour list is returned when there are no coords."""

    edges = [["A", 0.0, 0.0, 0.0]]  # Valid edge
    atom_labels = ["A"]
    neighbour_list = process_neighbour_list(edges, None, atom_labels)

    assert neighbour_list.shape == (0, 5)


def test_process_neighbour_list_empty_edges():
    """Test with empty edges list."""

    edges = []
    coords = np.array([[0.0, 0.0, 0.0]])
    atom_labels = ["A"]
    
    neighbour_list = process_neighbour_list(edges, coords, atom_labels)
    
    assert neighbour_list.shape == (0, 5)
