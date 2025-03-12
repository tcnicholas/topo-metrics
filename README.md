# topo-metrics

## Quickstart

```bash
# install Julia (if you haven't already)...
curl -fsSL https://install.julialang.org | sh

# install `topo-metrics` package using UV.
uv venv topo-throwaway --python 3.10
source topo-throwaway/bin/activate
uv pip install topo-metrics --upgrade
```

## Julia interface 

The Julia interface can sometimes take a bit of work to get up and running
(there is probably a way to improve the build...). If this doesn't work out of
the box, there are a couple of strategies I have found to help.

### 1. update PyCall.

```bash
python -c "import julia; julia.install()"
```

### 2. instantiate and precompile the project.

```bash

# grab the parent directory of the package.
PACKAGE_PARENT_DIR=$(python -c "import topo_metrics, os; print(os.path.dirname(topo_metrics.__file__))")

# access the Julia project.
cd ${PACKAGE_PARENT_DIR}/RingStatistics

# enter Julia
julia
```

Then activate the Julia Pkg manager using `]`, and run

```julia
instantiate
precompile
```