import os
import subprocess
import sys

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from hatchling.plugin import hookimpl


class CustomBuildHook(BuildHookInterface):
    """ Custom build hook to set up Julia before installation. """

    def initialize(self, version, build_data):
        """ Runs Julia setup before the package is installed. """

        # Get the active Python environment
        python_executable = sys.executable
        julia_package_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "RingStatistics")
        )

        print("🔧 Running Julia setup for topo-metrics...")

        # Step 1: Ensure Julia is updated.
        subprocess.run(["juliaup", "update"], check=True)

        # Step 2: Set the correct Python path in Julia’s PyCall.jl
        subprocess.run(
            [
                "julia", "--project=" + julia_package_path,
                "-e", (
                    f'using Pkg; ENV["PYTHON"]="{python_executable}";'
                    'Pkg.build("PyCall")'
                )
            ], check=True
        )

        # Step 3: Instantiate Julia project and precompile.
        subprocess.run(
            [
                "julia", "--project=" + julia_package_path, "-e", 
                "import Pkg; Pkg.instantiate(); Pkg.precompile()"
            ], 
            check=True
        )

        # Step 4: Ensure PyJulia is initialised properly.
        self.run_python_script_in_venv()

        print("✅ Julia setup completed successfully!")

    def run_python_script_in_venv(self):
        """ 
        Runs a small Python script inside the installed environment to run 
        julia.install() 
        """
        python_executable = sys.executable
        script = """if True:
            import julia
            julia.install()
            print("✅ PyJulia installed successfully in the environment!")
        """
        
        subprocess.run([python_executable, "-c", script], check=True)


@hookimpl
def hatch_register_build_hooks():
    """ Registers the custom Julia build hook in Hatch. """

    return {"custom": CustomBuildHook}
