from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure

from topo_metrics.analysis._style import pastelise
from topo_metrics.topology import RingsResults


def plot_ring_size_distributions(
    results: RingsResults | Sequence[RingsResults],
    *,
    labels: Sequence[str] | None = None,
    stacked: bool | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (4.0, 3.0),
    dpi: int = 150,
    width: float = 0.6,
    hatch: str | Sequence[str | None] | None = "//",
    pastel_amount: float = 0.55,
    edgecolor: (
        str
        | tuple[float, float, float]
        | tuple[float, float, float, float]
    ) = "k",
    xlabel: str = "ring size",
    ylabel: str = "count",
    xtick_step: int = 2,
    xlim: tuple[float, float] | None = (0.5, 15.0),
    legend: bool = True,
) -> tuple[Figure | SubFigure, Axes]:
    """
    Plot ring size distributions from one or multiple RingsResults.

    Parameters
    ----------
    results
        One instance or a list/tuple of instances. Each must have:
        results.ring_size_count.sizes and results.ring_size_count.counts
    labels
        Legend labels for each dataset. If None, uses "set 1", "set 2", ...
    stacked
        If None: stacked=True when multiple datasets, else False for single.
    ax
        Plot into an existing axes if provided.
    width
        Bar width.
    hatch
        Hatch pattern(s). If list, one per dataset. If None, no hatch.
    """

    def _looks_like_ringsresults(x):
        return (
            hasattr(x, "ring_size_count") and 
            hasattr(x.ring_size_count, "sizes") and 
            hasattr(x.ring_size_count, "counts")
        )

    if _looks_like_ringsresults(results):
        results_list = [results]
    else:
        results_list = list(results)

    n = len(results_list)
    if n == 0:
        raise ValueError("No RingsResults provided.")

    if stacked is None:
        stacked = (n > 1)

    if labels is None:
        labels = [f"set {i+1}" for i in range(n)]
    if len(labels) != n:
        raise ValueError(f"labels must have length {n} (got {len(labels)}).")

    all_sizes = sorted(
        set(np.concatenate([
            np.asarray(r.ring_size_count.sizes) # type: ignore
            for r in results_list 
        ]))
    )
    all_sizes = np.asarray(all_sizes)

    aligned_counts = []
    for r in results_list:
        assert isinstance(r, RingsResults)
        s = np.asarray(r.ring_size_count.sizes)
        c = np.asarray(r.ring_size_count.counts)
        d = dict(zip(s.tolist(), c.tolist())) # type: ignore
        aligned_counts.append(np.asarray([
            d.get(sz, 0) for sz in all_sizes], dtype=float)
        )

    base = plt.get_cmap("tab10").colors # type: ignore
    colors = [
        pastelise(base[i % len(base)], amount=pastel_amount) for i in range(n)
    ]

    if hatch is None:
        hatches = [None] * n
    elif isinstance(hatch, str):
        hatches = [hatch] * n
    else:
        hatches = list(hatch)
        if len(hatches) != n:
            raise ValueError(f"hatch must be a string or a list of length {n}.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    if stacked:
        bottom = np.zeros_like(all_sizes, dtype=float)
        for i in range(n):
            ax.bar(
                all_sizes,
                aligned_counts[i],
                bottom=bottom,
                color=colors[i],
                edgecolor=edgecolor,
                hatch=hatches[i] if hatches[i] else None,
                width=width,
                label=labels[i],
            )
            bottom = bottom + aligned_counts[i]
    else:
        if n == 1:
            offsets = np.array([0.0])
        else:
            group_span = min(0.85, 0.15 * n + 0.35)
            offsets = np.linspace(-group_span / 2, group_span / 2, n)

        for i in range(n):
            ax.bar(
                all_sizes + offsets[i],
                aligned_counts[i],
                color=colors[i],
                edgecolor=edgecolor,
                hatch=hatches[i] if hatches[i] else None,
                width=width / max(1, n) if n > 1 else width,
                label=labels[i] if n > 1 else None,
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Ticks / limits
    if len(all_sizes) > 0:
        ax.set_xticks(np.arange(2, int(np.max(all_sizes)) + 1, xtick_step))
    if xlim is not None:
        ax.set_xlim(*xlim)

    if legend and n > 1:
        ax.legend(frameon=False)

    return fig, ax