from pathlib import Path
from typing import Final

import pytest

from topo_metrics.io.cgd import parse_cgd
from topo_metrics.topology import Topology

DIA_FILE_PATH: Final = Path(__file__).parent / "data/alpha-cristobalite.cgd"


@pytest.fixture(scope="session")
def dia_file() -> Path:
    """Fixture that provides the path to the alpha-cristobalite.cgd file."""

    return DIA_FILE_PATH


@pytest.fixture
def parsed_cgd(dia_file: Path):
    """Fixture that parses the CGD file once and returns the results."""

    return parse_cgd(str(dia_file))


@pytest.fixture
def sample_topology(dia_file: Path):
    """Fixture that creates a Topology object from the CGD file."""

    return Topology.from_cgd(str(dia_file))


@pytest.fixture
def sample_converted_rnodes():
    """Fixture providing sample converted_rnodes for a 4-node dia-type net."""

    raw_rings = [
        [
            [
                (1, [0, 0, 0]),
                (4, [-1, 0, -1]),
                (3, [-1, -1, -1]),
                (2, [0, -1, -1]),
                (3, [0, -1, -1]),
                (4, [0, 0, -1]),
            ],
            [
                (1, [0, 0, 0]),
                (4, [-1, 0, -1]),
                (3, [-1, 0, -1]),
                (2, [0, 0, -1]),
                (3, [0, 0, -1]),
                (4, [0, 0, -1]),
            ],
            [
                (1, [0, 0, 0]),
                (2, [0, -1, 0]),
                (3, [0, -1, 0]),
                (2, [1, -1, 0]),
                (1, [1, 0, 0]),
                (4, [0, 0, -1]),
            ],
            [
                (1, [-1, 0, 0]),
                (2, [-1, -1, 0]),
                (3, [-1, -1, 0]),
                (2, [0, -1, 0]),
                (1, [0, 0, 0]),
                (4, [-1, 0, -1]),
            ],
            [
                (1, [0, 0, 0]),
                (2, [0, -1, 0]),
                (3, [0, -1, 0]),
                (4, [0, 0, 0]),
                (3, [0, 0, 0]),
                (2, [0, 0, 0]),
            ],
            [
                (1, [0, 0, 0]),
                (2, [0, -1, 0]),
                (3, [-1, -1, 0]),
                (4, [-1, 0, 0]),
                (3, [-1, 0, 0]),
                (2, [0, 0, 0]),
            ],
            [
                (1, [0, 0, 0]),
                (2, [0, 0, 0]),
                (1, [0, 1, 0]),
                (4, [0, 1, -1]),
                (3, [0, 0, -1]),
                (4, [0, 0, -1]),
            ],
            [
                (1, [0, -1, 0]),
                (2, [0, -1, 0]),
                (1, [0, 0, 0]),
                (4, [0, 0, -1]),
                (3, [0, -1, -1]),
                (4, [0, -1, -1]),
            ],
            [
                (1, [0, 0, 0]),
                (2, [0, 0, 0]),
                (3, [0, 0, 0]),
                (2, [1, 0, 0]),
                (1, [1, 0, 0]),
                (4, [0, 0, -1]),
            ],
            [
                (1, [-1, 0, 0]),
                (2, [-1, 0, 0]),
                (3, [-1, 0, 0]),
                (2, [0, 0, 0]),
                (1, [0, 0, 0]),
                (4, [-1, 0, -1]),
            ],
            [
                (1, [0, 0, 0]),
                (2, [0, 0, 0]),
                (1, [0, 1, 0]),
                (4, [-1, 1, -1]),
                (3, [-1, 0, -1]),
                (4, [-1, 0, -1]),
            ],
            [
                (1, [0, -1, 0]),
                (2, [0, -1, 0]),
                (1, [0, 0, 0]),
                (4, [-1, 0, -1]),
                (3, [-1, -1, -1]),
                (4, [-1, -1, -1]),
            ],
        ],
        [
            [
                (1, [0, 1, 1]),
                (4, [-1, 1, 0]),
                (3, [-1, 0, 0]),
                (2, [0, 0, 0]),
                (3, [0, 0, 0]),
                (4, [0, 1, 0]),
            ],
            [
                (1, [0, 0, 1]),
                (4, [-1, 0, 0]),
                (3, [-1, 0, 0]),
                (2, [0, 0, 0]),
                (3, [0, 0, 0]),
                (4, [0, 0, 0]),
            ],
            [
                (1, [0, 1, 0]),
                (2, [0, 0, 0]),
                (3, [0, 0, 0]),
                (2, [1, 0, 0]),
                (1, [1, 1, 0]),
                (4, [0, 1, -1]),
            ],
            [
                (1, [-1, 1, 0]),
                (2, [-1, 0, 0]),
                (3, [-1, 0, 0]),
                (2, [0, 0, 0]),
                (1, [0, 1, 0]),
                (4, [-1, 1, -1]),
            ],
            [
                (1, [0, 1, 0]),
                (2, [0, 0, 0]),
                (3, [0, 0, 0]),
                (4, [0, 1, 0]),
                (3, [0, 1, 0]),
                (2, [0, 1, 0]),
            ],
            [
                (1, [0, 0, 0]),
                (2, [0, -1, 0]),
                (3, [0, -1, 0]),
                (4, [0, 0, 0]),
                (3, [0, 0, 0]),
                (2, [0, 0, 0]),
            ],
            [
                (1, [0, 1, 0]),
                (2, [0, 0, 0]),
                (3, [-1, 0, 0]),
                (4, [-1, 1, 0]),
                (3, [-1, 1, 0]),
                (2, [0, 1, 0]),
            ],
            [
                (1, [0, 0, 0]),
                (2, [0, -1, 0]),
                (3, [-1, -1, 0]),
                (4, [-1, 0, 0]),
                (3, [-1, 0, 0]),
                (2, [0, 0, 0]),
            ],
            [
                (1, [0, 0, 0]),
                (2, [0, 0, 0]),
                (1, [0, 1, 0]),
                (4, [0, 1, -1]),
                (3, [0, 0, -1]),
                (4, [0, 0, -1]),
            ],
            [
                (1, [0, 0, 0]),
                (2, [0, 0, 0]),
                (3, [0, 0, 0]),
                (2, [1, 0, 0]),
                (1, [1, 0, 0]),
                (4, [0, 0, -1]),
            ],
            [
                (1, [-1, 0, 0]),
                (2, [-1, 0, 0]),
                (3, [-1, 0, 0]),
                (2, [0, 0, 0]),
                (1, [0, 0, 0]),
                (4, [-1, 0, -1]),
            ],
            [
                (1, [0, 0, 0]),
                (2, [0, 0, 0]),
                (1, [0, 1, 0]),
                (4, [-1, 1, -1]),
                (3, [-1, 0, -1]),
                (4, [-1, 0, -1]),
            ],
        ],
        [
            [
                (1, [1, 1, 1]),
                (4, [0, 1, 0]),
                (3, [0, 0, 0]),
                (2, [1, 0, 0]),
                (3, [1, 0, 0]),
                (4, [1, 1, 0]),
            ],
            [
                (1, [0, 1, 1]),
                (4, [-1, 1, 0]),
                (3, [-1, 0, 0]),
                (2, [0, 0, 0]),
                (3, [0, 0, 0]),
                (4, [0, 1, 0]),
            ],
            [
                (1, [1, 0, 1]),
                (4, [0, 0, 0]),
                (3, [0, 0, 0]),
                (2, [1, 0, 0]),
                (3, [1, 0, 0]),
                (4, [1, 0, 0]),
            ],
            [
                (1, [0, 0, 1]),
                (4, [-1, 0, 0]),
                (3, [-1, 0, 0]),
                (2, [0, 0, 0]),
                (3, [0, 0, 0]),
                (4, [0, 0, 0]),
            ],
            [
                (1, [0, 1, 0]),
                (2, [0, 0, 0]),
                (3, [0, 0, 0]),
                (2, [1, 0, 0]),
                (1, [1, 1, 0]),
                (4, [0, 1, -1]),
            ],
            [
                (1, [0, 1, 0]),
                (2, [0, 0, 0]),
                (3, [0, 0, 0]),
                (4, [0, 1, 0]),
                (3, [0, 1, 0]),
                (2, [0, 1, 0]),
            ],
            [
                (1, [0, 0, 0]),
                (2, [0, -1, 0]),
                (3, [0, -1, 0]),
                (4, [0, 0, 0]),
                (3, [0, 0, 0]),
                (2, [0, 0, 0]),
            ],
            [
                (1, [1, 1, 0]),
                (2, [1, 0, 0]),
                (3, [0, 0, 0]),
                (4, [0, 1, 0]),
                (3, [0, 1, 0]),
                (2, [1, 1, 0]),
            ],
            [
                (1, [1, 0, 0]),
                (2, [1, -1, 0]),
                (3, [0, -1, 0]),
                (4, [0, 0, 0]),
                (3, [0, 0, 0]),
                (2, [1, 0, 0]),
            ],
            [
                (1, [0, 0, 1]),
                (2, [0, 0, 1]),
                (1, [0, 1, 1]),
                (4, [0, 1, 0]),
                (3, [0, 0, 0]),
                (4, [0, 0, 0]),
            ],
            [
                (1, [0, 0, 0]),
                (2, [0, 0, 0]),
                (3, [0, 0, 0]),
                (2, [1, 0, 0]),
                (1, [1, 0, 0]),
                (4, [0, 0, -1]),
            ],
            [
                (1, [1, 0, 1]),
                (2, [1, 0, 1]),
                (1, [1, 1, 1]),
                (4, [0, 1, 0]),
                (3, [0, 0, 0]),
                (4, [0, 0, 0]),
            ],
        ],
        [
            [
                (1, [1, 0, 1]),
                (4, [0, 0, 0]),
                (3, [0, -1, 0]),
                (2, [1, -1, 0]),
                (3, [1, -1, 0]),
                (4, [1, 0, 0]),
            ],
            [
                (1, [0, 0, 1]),
                (4, [-1, 0, 0]),
                (3, [-1, -1, 0]),
                (2, [0, -1, 0]),
                (3, [0, -1, 0]),
                (4, [0, 0, 0]),
            ],
            [
                (1, [1, 0, 1]),
                (4, [0, 0, 0]),
                (3, [0, 0, 0]),
                (2, [1, 0, 0]),
                (3, [1, 0, 0]),
                (4, [1, 0, 0]),
            ],
            [
                (1, [0, 0, 1]),
                (4, [-1, 0, 0]),
                (3, [-1, 0, 0]),
                (2, [0, 0, 0]),
                (3, [0, 0, 0]),
                (4, [0, 0, 0]),
            ],
            [
                (1, [0, 0, 1]),
                (2, [0, -1, 1]),
                (3, [0, -1, 1]),
                (2, [1, -1, 1]),
                (1, [1, 0, 1]),
                (4, [0, 0, 0]),
            ],
            [
                (1, [0, 0, 0]),
                (2, [0, -1, 0]),
                (3, [0, -1, 0]),
                (4, [0, 0, 0]),
                (3, [0, 0, 0]),
                (2, [0, 0, 0]),
            ],
            [
                (1, [1, 0, 0]),
                (2, [1, -1, 0]),
                (3, [0, -1, 0]),
                (4, [0, 0, 0]),
                (3, [0, 0, 0]),
                (2, [1, 0, 0]),
            ],
            [
                (1, [0, -1, 1]),
                (2, [0, -1, 1]),
                (1, [0, 0, 1]),
                (4, [0, 0, 0]),
                (3, [0, -1, 0]),
                (4, [0, -1, 0]),
            ],
            [
                (1, [0, 0, 1]),
                (2, [0, 0, 1]),
                (1, [0, 1, 1]),
                (4, [0, 1, 0]),
                (3, [0, 0, 0]),
                (4, [0, 0, 0]),
            ],
            [
                (1, [0, 0, 1]),
                (2, [0, 0, 1]),
                (3, [0, 0, 1]),
                (2, [1, 0, 1]),
                (1, [1, 0, 1]),
                (4, [0, 0, 0]),
            ],
            [
                (1, [1, -1, 1]),
                (2, [1, -1, 1]),
                (1, [1, 0, 1]),
                (4, [0, 0, 0]),
                (3, [0, -1, 0]),
                (4, [0, -1, 0]),
            ],
            [
                (1, [1, 0, 1]),
                (2, [1, 0, 1]),
                (1, [1, 1, 1]),
                (4, [0, 1, 0]),
                (3, [0, 0, 0]),
                (4, [0, 0, 0]),
            ],
        ],
    ]

    return raw_rings
