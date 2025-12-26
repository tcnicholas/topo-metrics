from __future__ import annotations

import contextlib
import os
from pathlib import Path

os.environ.setdefault("PYTHON_JULIACALL_AUTOLOAD_IPYTHON_EXTENSION", "no")

import juliacall

jl = juliacall.newmodule("Rings")

RingStatistics = None


def get_project_root() -> Path:
    """Returns the project root directory."""

    return Path(__file__).resolve().parent


def get_project_src() -> Path:
    """Returns the path to the src directory located at the project root."""

    return get_project_root() / "src"


def get_data_dir() -> Path:
    """Returns the path to the data directory located at the project root."""

    return get_project_root() / "data"


def instantiate_julia() -> None:
    """
    Initialises Julia, activates the environment, and imports RingStatistics.
    """
    global RingStatistics

    from topo_metrics.paths import get_project_root

    rings_path = get_project_root() / "RingStatistics"

    # Initialise Julia interface
    jl.seval("import Pkg")

    # Activate RingStatistics Julia project
    jl.seval(f'''
        import Pkg
        redirect_stdout(devnull) do
            redirect_stderr(devnull) do
                Pkg.activate("{rings_path.as_posix()}")
            end
        end
    ''')

    # Load the RingStatistics module
    jl.seval(
        f'include("{(rings_path / "src" / "RingStatistics.jl").as_posix()}")'
    )

    # Set the global RingStatistics reference
    RingStatistics = jl.RingStatistics


with contextlib.suppress(ImportError):
    instantiate_julia()
