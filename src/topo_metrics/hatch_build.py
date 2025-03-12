import os
import subprocess
import sys

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from hatchling.plugin import hookimpl


class CustomBuildHook(BuildHookInterface):
    """ Custom build hook to set up Julia before installation. """

    def initialize(self, version, build_data):
        """ Runs Julia setup before the package is installed. """

        # get the active Python environment.
        python_executable = sys.executable
        julia_package_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "RingStatistics")
        )

        print("🔧 Running Julia setup for topo-metrics...")

        # step 1: Ensure Julia is updated.
        subprocess.run(["juliaup", "update"], check=True)

        # step 2: Set the correct Python path in Julia’s PyCall.jl
        subprocess.run(
            [
                "julia", "--project=" + julia_package_path,
                "-e", (
                    f'using Pkg; ENV["PYTHON"]="{python_executable}";'
                    'Pkg.build("PyCall")'
                )
            ], check=True
        )

        # step 3: Instantiate Julia project and precompile.
        subprocess.run(
            [
                "julia", "--project=" + julia_package_path, "-e", 
                "import Pkg; Pkg.instantiate(); Pkg.precompile()"
            ], 
            check=True
        )

        # step 4: Ensure PyJulia is initialized properly.
        import julia
        julia.install()

        print("✅ Julia setup completed successfully!")


@hookimpl
def hatch_register_build_hooks():
    """ Registers the custom Julia build hook in Hatch. """

    return {"custom": CustomBuildHook}
