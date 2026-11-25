"""Shared style of the data figures.

Serif text (Times New Roman), a muted categorical palette, a full black frame, a light
horizontal grid, black-edged bars with one hatch pattern per type of generation, value labels
on bars and frameless legends. Colour and hatch follow the type of generation, as in the
TikZ diagrams of the paper (blocks.tex, taxonomy.tex, primer.tex):

    autoregressive (and Chronos)          blue    //
    direct multi-step (and Chronos-Bolt)  orange  --
    hybrid (and Sundial)                  green   \\
    full-window diffusion                 khaki   ..
    full-window flow matching             red     (solid)
    masked diffusion                      purple  xx

Usage:  import figstyle; figstyle.apply()
"""
import colorsys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

BLUE, ORANGE, GREEN, RED = "#4C72B0", "#DD8452", "#55A868", "#C44E52"
PURPLE, BROWN, PINK, GREY, KHAKI, CYAN = "#8172B2", "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"

MODEL_COLOR = {"ar": BLUE, "chronos": BLUE, "direct": ORANGE, "bolt_ind": ORANGE, "bolt_com": ORANGE,
               "ardiff": GREEN, "sundial": GREEN, "fulldiff": KHAKI, "flow": RED,
               "masked": PURPLE, "masked_rand": PURPLE,
               # further foundation models, by type of generation
               "toto": BLUE, "timesfm_ind": BLUE, "timesfm_com": BLUE,
               "chronos2_ind": ORANGE, "chronos2_com": ORANGE, "chronos2_native_ind": ORANGE,
               "aurora": RED, "aurora_tok24": RED}
MODEL_HATCH = {"ar": "//", "chronos": "//", "direct": "--", "bolt_ind": "--", "bolt_com": "--",
               "ardiff": "\\\\", "sundial": "\\\\", "fulldiff": "..", "flow": "",
               "masked": "xx", "masked_rand": "xx",
               "toto": "//", "timesfm_ind": "//", "timesfm_com": "//",
               "chronos2_ind": "--", "chronos2_com": "--", "chronos2_native_ind": "--",
               "aurora": "", "aurora_tok24": ""}
INK, INK2, MUTED, GRID, AXIS = "#000000", "#262626", "#7f7f7f", "#e6e6e6", "#b3b3b3"
FRAME = 0.6                    # width (pt) of axes frames, tick marks and bar outlines


def apply():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.linewidth": FRAME,
        "xtick.major.width": FRAME, "ytick.major.width": FRAME, "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.spines.top": True, "axes.spines.right": True,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.titleweight": "bold", "axes.titlelocation": "center",
        "legend.frameon": False, "legend.handlelength": 1.6,
        "hatch.linewidth": 0.6, "hatch.color": INK,
        "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "savefig.dpi": 300,
    })


def darker(color, factor=0.72):
    """A darker shade of a colour (for a second series of the same type)."""
    h, l, s = colorsys.rgb_to_hls(*mcolors.to_rgb(color))
    return colorsys.hls_to_rgb(h, l * factor, s)


def lighter(color, factor=0.45):
    """A lighter tint of a colour (mixed with white)."""
    r, g, b = mcolors.to_rgb(color)
    return (r + (1 - r) * factor, g + (1 - g) * factor, b + (1 - b) * factor)


def patch(color, hatch, label):
    """Legend handle for a hatched bar."""
    from matplotlib.patches import Patch
    return Patch(facecolor=color, edgecolor=INK, hatch=hatch, linewidth=FRAME, label=label)


def text_on(color):
    """Black or white text, whichever reads better on the given fill."""
    r, g, b = mcolors.to_rgb(color)
    return INK if 0.299 * r + 0.587 * g + 0.114 * b > 0.6 else "white"


def bars(ax, x, heights, width, color, hatch="", yerr=None, labels=None, fmt="{:.3f}", label=None,
         fontsize=6.5, rotation=0, pad=1.5):
    """Black-edged, hatched bars with optional error bars and value labels above them."""
    b = ax.bar(x, heights, width, color=color, edgecolor=INK, linewidth=FRAME, hatch=hatch, label=label,
               zorder=2)
    errs = [0.0] * len(heights) if yerr is None else list(yerr)
    if yerr is not None:
        ax.errorbar(x, heights, yerr=yerr, fmt="none", ecolor=INK, elinewidth=FRAME, capsize=1.5, zorder=3)
    if labels is not False:
        texts = labels if labels is not None else [fmt.format(h) for h in heights]
        texts = [t[1:] if t.startswith("-") and float(t) == 0 else t for t in texts]     # no "-0"
        for xi, h, e, t in zip(x, heights, errs, texts):
            up = h >= 0
            ax.annotate(t, (xi, h + e if up else h - e), xytext=(0, pad if up else -pad),
                        textcoords="offset points", ha="center", va="bottom" if up else "top",
                        fontsize=fontsize, rotation=rotation, zorder=4)
    return b
