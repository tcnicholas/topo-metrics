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

## Documentation

Checkout the [documentation](https://tcnicholas.github.io/topo-metrics/) for 
some more detailed examples of the available functionality.

(This is in working progress...)

## Citation

If you find `topo-metrics` helpful for your research, please cite the preprint 
in which we introduced the `CARVS` notation:

```bibtex
@misc{nicholas2025amof,
  title         = {The structure and topology of an amorphous metal-organic framework},
  author        = {Nicholas, Thomas C. and Thomas du Toit, Daniel F. and Rosset, Louise A. M. and Proserpio, Davide M. and Goodwin, Andrew L. and Deringer, Volker L.},
  year          = {2025},
  month         = mar,
  eprint        = {2503.24367},
  archivePrefix = {arXiv},
  primaryClass  = {cond-mat.mtrl-sci},
  doi           = {10.48550/arXiv.2503.24367},
  url           = {https://arxiv.org/abs/2503.24367}
}
```

`topo-metrics` relies on 
[`PeriodicGraphs.jl`](https://github.com/Liozou/PeriodicGraphs.jl) and the 
topology-identification routines provided by 
[`CrystalNets.jl`](https://github.com/coudertlab/CrystalNets.jl). If you use 
`topo-metrics` in published work, please also cite the `CrystalNets.jl` 
companion article:

```bibtex
@article{zoubritzky2022crystalnets,
  title   = {CrystalNets.jl: Identification of Crystal Topologies},
  author  = {Zoubritzky, Lionel and Coudert, Fran{\c{c}}ois-Xavier},
  journal = {SciPost Chem.},
  volume  = {1},
  number  = {2},
  pages   = {005},
  year    = {2022},
  doi     = {10.21468/SciPostChem.1.2.005},
  url     = {https://scipost.org/SciPostChem.1.2.005}
}
```