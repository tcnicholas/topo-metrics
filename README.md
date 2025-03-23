# topo-metrics

`topo-metrics` provides topological analysis tools for network materials.

<div align="center">

![PyPI - Version](https://img.shields.io/pypi/v/topo-metrics)
[![GitHub License](https://img.shields.io/github/license/tcnicholas/topo-metrics)](LICENSE.md)
[![](https://github.com/tcnicholas/topo-metrics/actions/workflows/tests.yaml/badge.svg?branch=main)](https://github.com/tcnicholas/topo-metric/actions/workflows/tests.yaml)
[![codecov](https://codecov.io/gh/tcnicholas/topo-metrics/graph/badge.svg?token=S3K9PSK7NA)](https://codecov.io/gh/tcnicholas/topo-metrics)
</div>

---

## 🚀 Quickstart

### 1. Install Julia

You'll need a working [Julia installation](https://julialang.org/downloads/):

**On Linux/macOS:**
```bash
curl -fsSL https://install.julialang.org | sh
```

**On Windows (via winget):**
```bash
winget install julia -s msstore
```

### 2. Set up Python environment (Python ≥ 3.10)

We recommend using conda or UV to manage environments:

```bash
conda create -n topo-env python=3.10 -y
conda activate topo-env
pip install topo-metrics
```

### 3. First run initialisation

On the **first run**, `topo-metrics` will: 

- Set up the Julia runtime environment.
- Resolve dependencies and precompile Julia code.
- Create a persistent link between Python and Julia using [PythonCall.jl](https://juliapy.github.io/PythonCall.jl/stable/).

This process only happens once—future runs should be fast and seamless.

### ✅ You're ready to go!

Try running one of the example scripts:

```bash
python examples/main.py alpha-cristobalite
```

Or check out the [examples directory](./examples).
