"""Setup Matplotlib defaults with LaTeX rendering
for quality plots"""

import matplotlib as mpl

mpl.rcParams.update({
    # use LaTeX for text
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "mathtext.fontset": "cm",

    # figure
    "figure.figsize": (4, 3),  # inches
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.autolayout": True,

    # axes
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.7,

    # ticks
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 5,
    "ytick.major.size": 5,

    # lines
    "lines.linewidth": 1.0,
    "lines.markersize": 3,

    # legend
    "legend.fontsize": 8,

    # color cycle
    "axes.prop_cycle": mpl.cycler(
        "color",
        [
            "#1f77b4", "#ff7f0e", "#2ca02c",
            "#d62728", "#9467bd", "#8c564b",
            "#e377c2", "#7f7f7f", "#bcbd22",
            "#17becf",
        ],
    ),
})
