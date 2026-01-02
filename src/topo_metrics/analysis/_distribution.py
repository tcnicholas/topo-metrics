from __future__ import annotations

from typing import Literal, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure

from topo_metrics.analysis._style import pastelise
from topo_metrics.topology import RingsResults

WrithePlotKind = Literal[
    "stacked_hist", "hist", "box", "violin", "mean", "scatter"
]


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
        str | tuple[float, float, float] | tuple[float, float, float, float]
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
            hasattr(x, "ring_size_count")
            and hasattr(x.ring_size_count, "sizes")
            and hasattr(x.ring_size_count, "counts")
        )

    if _looks_like_ringsresults(results):
        results_list = [results]
    else:
        results_list = list(results)

    n = len(results_list)
    if n == 0:
        raise ValueError("No RingsResults provided.")

    if stacked is None:
        stacked = n > 1

    if labels is None:
        labels = [f"set {i + 1}" for i in range(n)]
    if len(labels) != n:
        raise ValueError(f"labels must have length {n} (got {len(labels)}).")

    all_sizes = sorted(
        set(
            np.concatenate(
                [
                    np.asarray(r.ring_size_count.sizes)  # type: ignore
                    for r in results_list
                ]
            )
        )
    )
    all_sizes = np.asarray(all_sizes)

    aligned_counts = []
    for r in results_list:
        assert isinstance(r, RingsResults)
        s = np.asarray(r.ring_size_count.sizes)
        c = np.asarray(r.ring_size_count.counts)
        d = dict(zip(s.tolist(), c.tolist()))  # type: ignore
        aligned_counts.append(
            np.asarray([d.get(sz, 0) for sz in all_sizes], dtype=float)
        )

    base = plt.get_cmap("tab10").colors  # type: ignore
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


def plot_writhe_distributions(
    writhes_data: Mapping[int, Sequence[float]] | Sequence[float] | np.ndarray,
    *,
    kind: WrithePlotKind = "stacked_hist",
    bins: int | Sequence[float] | npt.NDArray[np.floating] = 30,
    stacked: bool | None = None,
    absval: bool = True,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (4.0, 3.0),
    dpi: int = 150,
    colour_map: str = "viridis",
    pastel_amount: float = 0.55,
    edgecolor: (
        str | tuple[float, float, float] | tuple[float, float, float, float]
    ) = "k",
    linewidth: float = 1.0,
    alpha: float = 1.0,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend: bool = True,
    xtick_step: int = 2,
    xlim: tuple[float, float] | None = None,
    min_count: int = 1,
) -> tuple[Figure | SubFigure, Axes]:
    """
    Plot writhe distributions as a function of ring size.

    Parameters
    ----------
    writhes_data
        If dict: {ring_size: writhe_values}.
        If list/array: treated as all values concatenated (single distribution).
    kind
        - "stacked_hist": histogram stacked by ring size (dict input only).
        - "hist": overlaid histograms by ring size (dict) or single (array).
        - "box": boxplots of writhe vs ring size.
        - "violin": violin plots of writhe vs ring size.
        - "mean": mean ± SEM of writhe vs ring size.
        - "scatter": raw points of writhe vs ring size.
    bins
        Histogram bins (int or explicit bin edges).
    stacked
        For histogram kinds: if None, defaults to True for "stacked_hist",
        else False.
    absval
        If True, plot $|W_r|$ (default).
    min_count
        Skip ring sizes with fewer than this many values.

    Returns
    -------
    fig, ax
    """

    is_dict = isinstance(writhes_data, Mapping)

    if stacked is None:
        stacked = kind == "stacked_hist"

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    # --- normalise input into either:
    # (A) flat array "values"
    # (B) dict {size: array(values)}
    if not is_dict:
        vals = np.asarray(writhes_data, dtype=float).ravel()
        if absval:
            vals = np.abs(vals)

        if kind in ("box", "violin", "mean", "scatter"):
            raise ValueError(
                f"kind='{kind}' "
                "requires dict input {ring_size: values}. "
                "For a flat list/array, use kind='hist' or kind='stacked_hist'."
            )

        ax.hist(
            vals,
            bins=bins,  # type: ignore
            stacked=False,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
            label=None,
        )

        ax.set_xlabel(xlabel or ("$|W_r|$" if absval else "writhe"))
        ax.set_ylabel(ylabel or "count")
        if xlim is not None:
            ax.set_xlim(*xlim)
        return fig, ax

    data_dict: dict[int, np.ndarray] = {}
    for k, v in writhes_data.items():
        arr = np.asarray(v, dtype=float).ravel()
        if absval:
            arr = np.abs(arr)
        if arr.size >= min_count:
            data_dict[int(k)] = arr

    if len(data_dict) == 0:
        raise ValueError(
            "No writhe data to plot (all entries empty / below min_count)."
        )

    sizes = np.array(sorted(data_dict.keys()), dtype=int)

    cmap = plt.get_cmap(colour_map)
    smin, smax = float(sizes.min()), float(sizes.max())
    colors = [
        pastelise(
            cmap((s - smin) / max(1.0, smax - smin)), amount=pastel_amount
        )
        for s in sizes
    ]

    # --- plot by kind ---
    if kind in ("stacked_hist", "hist"):
        series = [data_dict[s] for s in sizes]
        labels = [f"n={int(s)}" for s in sizes]

        ax.hist(
            series,
            bins=bins,  # type: ignore
            stacked=bool(stacked),
            label=labels if legend else None,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
            color=colors,
        )

        ax.set_xlabel(xlabel or ("$|W_r|$" if absval else "writhe"))
        ax.set_ylabel(ylabel or "count")
        if legend:
            ax.legend(frameon=False)

        if xlim is not None:
            ax.set_xlim(*xlim)
        return fig, ax

    if kind == "box":
        series = [data_dict[s] for s in sizes]
        bp = ax.boxplot(
            series,
            positions=sizes,
            widths=0.6,
            patch_artist=True,
            manage_ticks=False,
            showfliers=False,
        )
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_edgecolor(edgecolor)
            patch.set_linewidth(linewidth)

        for key in ("whiskers", "caps", "medians"):
            for line in bp[key]:
                line.set_color(edgecolor)
                line.set_linewidth(linewidth)

        ax.set_xlabel(xlabel or "ring size")
        ax.set_ylabel(ylabel or ("$|W_r|$" if absval else "writhe"))
        ax.set_xticks(np.arange(sizes.min(), sizes.max() + 1, xtick_step))
        if xlim is not None:
            ax.set_xlim(*xlim)
        return fig, ax

    if kind == "violin":
        series = [data_dict[s] for s in sizes]
        vp = ax.violinplot(
            series,
            positions=sizes,
            widths=0.8,
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )
        for body, c in zip(vp["bodies"], colors):  # type: ignore
            body.set_facecolor(c)
            body.set_edgecolor(edgecolor)
            body.set_linewidth(linewidth)
            body.set_alpha(alpha)

        # medians line
        if "cmedians" in vp:
            vp["cmedians"].set_color(edgecolor)
            vp["cmedians"].set_linewidth(linewidth)

        ax.set_xlabel(xlabel or "ring size")
        ax.set_ylabel(ylabel or ("$|W_r|$" if absval else "writhe"))

        ax.set_xticks(np.arange(sizes.min(), sizes.max() + 1, xtick_step))
        if xlim is not None:
            ax.set_xlim(*xlim)
        return fig, ax

    if kind == "mean":
        means = np.array([data_dict[s].mean() for s in sizes], dtype=float)
        sems = np.array(
            [
                data_dict[s].std(ddof=1) / np.sqrt(max(1, data_dict[s].size))
                for s in sizes
            ],
            dtype=float,
        )

        ax.errorbar(
            sizes,
            means,
            yerr=sems,
            fmt="o-",
            color=edgecolor,
            linewidth=linewidth,
            markersize=4,
            capsize=3,
        )

        ax.set_xlabel(xlabel or "ring size")
        ax.set_ylabel(ylabel or ("mean $|W_r|$" if absval else "mean writhe"))
        ax.set_xticks(np.arange(sizes.min(), sizes.max() + 1, xtick_step))
        if xlim is not None:
            ax.set_xlim(*xlim)
        return fig, ax

    if kind == "scatter":
        for i, s in enumerate(sizes):
            y = data_dict[s]
            x = np.full_like(y, float(s))
            ax.scatter(
                x,
                y,
                s=8,
                alpha=0.6,
                color=colors[i],
                edgecolors="none",
                label=f"n={int(s)}" if legend else None,
            )

        ax.set_xlabel(xlabel or "ring size")
        ax.set_ylabel(ylabel or ("$|W_r|$" if absval else "$W_r$"))
        ax.set_xticks(np.arange(sizes.min(), sizes.max() + 1, xtick_step))
        if legend:
            ax.legend(frameon=False, ncol=2)
        if xlim is not None:
            ax.set_xlim(*xlim)
        return fig, ax

    raise ValueError(f"Unknown kind: {kind!r}")
