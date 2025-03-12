from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pytest
from pymatgen.core import Lattice as PymatgenLattice

from src.topo_metrics.io.cgd import parse_cgd, process_neighbour_list

DIA_FILE: Final = Path(__file__).parent / "data/alpha-cristobalite.cgd"

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
    ['Si1',  0.8005,  0.1995, -0.25],
    ['Si1', -0.1995,  0.1995, -0.25],
    ['Si1',  0.1995, -0.1995,  0.25],
    ['Si1',  0.1995,  0.8005,  0.25],
    ['Si2', -0.3005,  0.6995,  0.5],
    ['Si2',  0.3005,  1.3005,  0.0],
    ['Si2',  0.6995,  0.6995,  0.5],
    ['Si2',  0.3005,  0.3005,  0.0],
    ['Si3',  0.8005,  0.1995,  0.75],
    ['Si3',  1.1995,  0.8005,  0.25],
    ['Si3',  0.8005,  1.1995,  0.75],
    ['Si3',  0.1995,  0.8005,  0.25],
    ['Si4',  0.6995, -0.3005,  0.5],
    ['Si4',  0.6995,  0.6995,  0.5],
    ['Si4',  1.3005,  0.3005,  1.0],
    ['Si4',  0.3005,  0.3005,  1.0],
]

TRUTH_NEIGHBOUR_LIST: Final[npt.NDArray[np.int_]] = np.array([
    [ 1,  4,  0,  0, -1],
    [ 1,  4, -1,  0, -1],
    [ 1,  2,  0, -1,  0],
    [ 1,  2,  0,  0,  0],
    [ 2,  3, -1,  0,  0],
    [ 2,  1,  0,  1,  0],
    [ 2,  3,  0,  0,  0],
    [ 2,  1,  0,  0,  0],
    [ 3,  4,  0,  0,  0],
    [ 3,  2,  1,  0,  0],
    [ 3,  4,  0,  1,  0],
    [ 3,  2,  0,  0,  0],
    [ 4,  3,  0, -1,  0],
    [ 4,  3,  0,  0,  0],
    [ 4,  1,  1,  0,  1],
    [ 4,  1,  0,  0,  1]
])


@pytest.fixture
def parsed_cgd():
    """ Fixture that parses the CGD file once and returns the results. """
    return parse_cgd(DIA_FILE)


def test_parse_cgd(parsed_cgd):
    lattice, atom_labels, all_coords, edges = parsed_cgd
    assert isinstance(lattice, PymatgenLattice)
    assert lattice == TRUTH_LATTICE
    assert atom_labels == TRUTH_ATOM_LABELS
    np.testing.assert_allclose(all_coords, TRUTH_COORDS)
    assert edges == TRUTH_EDGES


def test_neighbour_list(parsed_cgd):
    _, atom_labels, all_coords, edges = parsed_cgd
    neighbour_list = process_neighbour_list(edges, all_coords, atom_labels)
    np.testing.assert_equal(neighbour_list, TRUTH_NEIGHBOUR_LIST)