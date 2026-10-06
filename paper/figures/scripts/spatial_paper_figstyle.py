"""Shared figure style for the manuscript.

Nature Methods print conventions: sans-serif, 5-7 pt type, 89 mm single-column
or 183 mm double-column width, thin marks, recessive axes.

Palette provenance: the two-colour categorical pair was validated with the
dataviz palette validator and passes every check on a light surface --
lightness band, chroma floor, CVD separation (worst adjacent pair dE 33.8
protan / 29.1 tritan), normal-vision floor (dE 40.7) and contrast >= 3:1.
Do not substitute colours without re-running that validator.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# --- canvas -----------------------------------------------------------------
MM = 1 / 25.4
SINGLE_COL = 89 * MM
DOUBLE_COL = 183 * MM

# --- palette (validated; see module docstring) ------------------------------
STAR = "#0F62FE"        # open / molecule-first fields
SR = "#D97706"          # Space Ranger fields
INK = "#1a1a1a"         # primary text
INK_MUTED = "#6b7280"   # secondary text, axis labels
GRID = "#e5e7eb"
CHANCE = "#374151"      # reference lines

# Policy shades, used only where a panel must distinguish hard from soft.
# Same hue, different lightness: policy is a magnitude-like nuisance dimension,
# not a second identity, so it must not consume a categorical hue.
STAR_SOFT = "#7CA9FF"
SR_SOFT = "#F0B457"


def apply():
    """Install the manuscript rcParams. Call once at the top of a figure script."""
    mpl.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,          # embed TrueType so text stays editable
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 6.5,
        "axes.labelsize": 6.5,
        "axes.titlesize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.6,
        "axes.edgecolor": INK_MUTED,
        "axes.labelcolor": INK,
        "axes.titlelocation": "left",
        "axes.titlepad": 4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "lines.linewidth": 1.0,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.2,
    })


def panel_label(ax, letter, dx=-0.19, dy=1.10):
    """Bold lower-case panel letter, Nature style."""
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="top", ha="left", color=INK)


def hide_y_grid_behind(ax):
    """Recessive horizontal grid, drawn behind the marks."""
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
