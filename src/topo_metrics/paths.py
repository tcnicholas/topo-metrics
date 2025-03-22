from __future__ import annotations

import contextlib
from pathlib import Path

import juliacall

jl = juliacall.newmodule("Rings")  # avoid polluting the global namespace.

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

    # get the path to the RingStatistics module.
    ring_stats_path = get_project_root() / "RingStatistics"

    # initialise Julia interface.
    jl.seval("import Pkg")

    jl.seval(f'''
        import Pkg
        redirect_stdout(devnull) do
            redirect_stderr(devnull) do
                Pkg.activate("{ring_stats_path}")
            end
        end
    ''')

    # import the RingStatistics module.
    jl.seval(f'include("{ring_stats_path / "src" / "RingStatistics.jl"}")')

    # assign the RingStatistics module to the global variable.
    RingStatistics = jl.RingStatistics


with contextlib.suppress(ImportError):
    instantiate_julia()
