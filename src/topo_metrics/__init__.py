import os
import subprocess
import sys

from topo_metrics.topology import Topology

__all__ = ["Topology"]
__version__ = "0.0.10"


def _setup_julia():
    """ Set up Julia dependencies after installation (only once). """

    julia_package_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "RingStatistics")
    )
    flag_file = os.path.join(julia_package_path, ".julia_setup_done")

    # If setup has already run, skip it
    if os.path.exists(flag_file):
        return

    python_executable = sys.executable

    try:
        print("🔧 Running Julia setup (this will only run once)...")

        # Ensure Julia is updated.
        subprocess.run(["juliaup", "update"], check=True)

        # Set Python path in Julia’s PyCall.jl
        subprocess.run(
            [
                "julia", "--project=" + julia_package_path, "-e", 
                (
                    f'using Pkg; ENV["PYTHON"]="{python_executable}"; ' 
                    'Pkg.build("PyCall")'
                )
            ], check=True
        )

        # Instantiate and precompile the Julia project
        subprocess.run(
            [
                "julia", "--project=" + julia_package_path,
                "-e", "import Pkg; Pkg.instantiate(); Pkg.precompile()"
            ], check=True
        )

        # Create the flag file to indicate setup is complete
        with open(flag_file, "w") as f:
            f.write("Julia setup completed")

        print("✅ Julia setup completed successfully!")

    except subprocess.CalledProcessError as e:
        print(f"⚠️ Julia setup failed: {e}")

# Run the setup only if needed
_setup_julia()
