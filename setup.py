import os
import subprocess
import sys

from setuptools import setup
from setuptools.command.install import install


class PostInstallCommand(install):
    """ Post-installation for setting up Julia dependencies. """
    
    def run(self):
        install.run(self)  # Run standard install process
        print("🔧 Running Julia setup after installation...")

        python_executable = sys.executable
        julia_package_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), 
                "src", "topo_metrics", "RingStatistics"
            )
        )

        try:
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

            # Step 4: Ensure PyJulia is initialized properly.
            self.run_python_script_in_venv()

            print("✅ Julia setup completed successfully!")

        except subprocess.CalledProcessError as e:
            print(f"⚠️ Julia setup failed: {e}")
            sys.exit(1)

    def run_python_script_in_venv(self):
        """
        Runs a small Python script inside the installed environment to run 
        julia.install().
        """
        python_executable = sys.executable
        script = """if True:
            import julia
            julia.install()
            print("✅ PyJulia installed successfully in the environment!")
        """

        subprocess.run([python_executable, "-c", script], check=True)


setup(
    name="topo-metrics",
    version="0.0.7",
    install_requires=[
        "setuptools",
        "wheel",
        "julia>=0.6.2",
        "pymatgen>=2025.3.10",
        "pytest>=8.3.5",
    ],
    cmdclass={ "install": PostInstallCommand },
)
