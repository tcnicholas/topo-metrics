# topo-metrics

<div align="center">

![PyPI - Version](https://img.shields.io/pypi/v/topo-metrics)
[![GitHub License](https://img.shields.io/github/license/tcnicholas/topo-metrics)](LICENSE.md)
[![](https://github.com/tcnicholas/topo-metrics/actions/workflows/tests.yaml/badge.svg?branch=main)](https://github.com/tcnicholas/topo-metrics/actions/workflows/tests.yaml)
[![codecov](https://codecov.io/gh/tcnicholas/topo-metrics/graph/badge.svg?token=S3K9PSK7NA)](https://codecov.io/gh/tcnicholas/topo-metrics)
</div>


## Install

```bash
# install Julia (if you haven't already)...
curl -fsSL https://install.julialang.org | sh

# install `topo-metrics` package using UV.
uv venv topo-throwaway --python 3.10
source topo-throwaway/bin/activate
uv pip install topo-metrics
```

## Quickstart

```pycon
>>> from ase.io import read
>>> import topo_metrics as tm
>>>
>>> atoms = read("zeolite-sodalite.cif")
>>> graph = tm.Topology.from_ase(ase_atoms=atoms, cutoff=1.7, remove_types={"O"})
>>> ring_stats = graph.get_clusters()
>>> ring_stats

RingsResults(
  depth=12,
  strong_rings=False,
  ring_size_count=RingSizeCounts(n_rings=46, min=4, max=12),
  VertexSymbol=[4.4.6.6.6.6],
  CARVS={4(2).6(4).12(32)}
)
```

## Docs

Checkout the [documentation](https://tcnicholas.github.io/topo-metrics/) for 
some more detailed examples of the available functionality.