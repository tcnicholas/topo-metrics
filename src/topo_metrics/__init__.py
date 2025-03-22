from __future__ import annotations

import os
import subprocess

from topo_metrics.topology import Topology

__all__ = ["Topology"]
__version__ = "0.0.12"


def _setup_julia():
    """Set up Julia project: instantiate and precompile using PythonCall."""

    julia_package_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "RingStatistics")
    )

    flag_file = os.path.join(julia_package_path, ".julia_setup_done")

    if os.path.exists(flag_file):
        return

    try:
        print("🔧 Running Julia setup (this will only run once)...")

        # Precompile Julia project
        subprocess.run(
            [
                "julia",
                f"--project={julia_package_path}",
                "-e",
                "using Pkg; Pkg.resolve(); Pkg.instantiate(); Pkg.precompile()",
            ],
            check=True,
        )

        with open(flag_file, "w") as f:
            f.write("Julia setup completed")

        print("✅ Julia setup completed successfully!")

    except subprocess.CalledProcessError as e:
        print(f"⚠️ Julia setup failed: {e}")


_setup_julia()
