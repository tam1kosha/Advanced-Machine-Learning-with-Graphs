"""Shared figure style: one readable name and one fixed colour per method, quiet axes.

Use it in every new figure so a method keeps its colour and its name across the repo:

    from pmgraph.plotstyle import METHOD_COLOR, METHOD_LABEL, apply_style
    apply_style()
    ax.plot(k, y, color=METHOD_COLOR["hgnn"], label=METHOD_LABEL["hgnn"])

The four hues are colour-blind safe as a set (checked on a light background). The non-learned
baseline is grey on purpose: it is the reference every learned model has to clear.
"""
import matplotlib as mpl

METHOD_LABEL = {
    "hist": "history-only (no training)",
    "xgb_own": "XGBoost, own trades",
    "xgb_nbr": "XGBoost + neighbour averages",
    "spectral_lr": "spectral embedding + LR",
    "labelprop": "label propagation",
    "sage": "GraphSAGE (counterparties)",
    "hgnn": "hetero GraphSAGE (wallets + markets)",
}
METHOD_COLOR = {
    "hist": "#8a8984",
    "xgb_own": "#2a78d6",
    "xgb_nbr": "#eb6834",
    "sage": "#1baf7a",
    "hgnn": "#6250d6",
    "spectral_lr": "#e87ba4",
    "labelprop": "#eda100",
}
BASELINES, GRAPH_METHODS = ["hist", "xgb_own"], ["xgb_nbr", "sage", "hgnn"]

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
ACCENT, WARM = "#2a78d6", "#e34948"   # "the thing we recommend" / "the thing it replaces"


def apply_style() -> None:
    """Quiet defaults: light surface, hairline solid grid, no top/right spines, 2 px lines."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.linewidth": 1.0, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.axisbelow": True, "axes.labelcolor": INK2, "axes.titlecolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 0, "ytick.major.size": 0,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 9.5, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
        "legend.frameon": False, "lines.linewidth": 2.0, "lines.markersize": 6.5,
        "lines.markeredgecolor": SURFACE, "lines.markeredgewidth": 1.5, "savefig.dpi": 160,
    })
