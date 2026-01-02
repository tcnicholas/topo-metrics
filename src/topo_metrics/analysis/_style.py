from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import to_rgb

mpl.rcParams["hatch.linewidth"] = 0.8


def pastelise(color, amount: float = 0.55) -> tuple[float, float, float]:
    """Blend color toward white by `amount` (0=original, 1=white)."""

    r, g, b = to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)
