"""Shared figure style for the manuscript.

Every figure is drawn at the width of the manuscript's text block and saved at
exactly that width (see save), so the type prints at the sizes set here and is
the same in every figure: sans-serif, 6 to 9 pt, thin marks, recessive axes.
Nothing is set below FS_NOTE.

Palette provenance: the two-colour categorical pair was validated with the
dataviz palette validator and passes every check on a light surface --
lightness band, chroma floor, CVD separation (worst adjacent pair dE 33.8
protan / 29.1 tritan), normal-vision floor (dE 40.7) and contrast >= 3:1.
Do not substitute colours without re-running that validator.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

# --- canvas -----------------------------------------------------------------
MM = 1 / 25.4
TEXT_WIDTH = 451.3 / 72     # inches: A4 with 1 in margins, the manuscript's \linewidth

# --- type sizes, in points as printed -------------------------------------------
FS_PANEL = 9       # panel letters
FS_TITLE = 7.5     # panel titles
FS_LABEL = 7       # axis labels and running text
FS_TICK = 6.5      # tick labels and legends
FS_NOTE = 6        # value labels and notes; the smallest size used

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
        "font.size": FS_LABEL,
        "axes.labelsize": FS_LABEL,
        "axes.titlesize": FS_TITLE,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_TICK,
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
    """Bold capital panel letter (Genome Research style); the letter may be given in either case."""
    ax.text(dx, dy, letter.upper(), transform=ax.transAxes,
            fontsize=FS_PANEL, fontweight="bold", va="top", ha="left", color=INK)


def save(fig, out_dir, stem, pad=0.03):
    """Write PDF and PNG at exactly the figure's width; crop the height to the content.

    The manuscript sets each figure at the text width, so a file of that width is
    printed without scaling. Stops if anything is drawn outside the canvas, since
    a fixed width cannot grow to hold it.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [fig.get_tightbbox(renderer)]
    # The tight box leaves out the length of axis labels, so add them.
    for ax in fig.axes:
        for label in (ax.xaxis.label, ax.yaxis.label):
            if ax.axison and label.get_visible() and label.get_text().strip():
                boxes.append(Bbox(label.get_window_extent(renderer).get_points() / fig.dpi))
    tight = Bbox.union(boxes)
    width = fig.get_figwidth()
    if tight.x0 < 0 or tight.x1 > width:
        raise SystemExit(f"{stem}: content spans {tight.x0:.3f} to {tight.x1:.3f} in, "
                         f"outside the {width:.3f} in canvas")
    box = Bbox.from_extents(0, tight.y0 - pad, width, tight.y1 + pad)
    for ext in ("pdf", "png"):
        fig.savefig(Path(out_dir) / f"{stem}.{ext}", bbox_inches=box)
    print(f"wrote {Path(out_dir) / stem}.pdf: content {tight.x0:.2f}-{tight.x1:.2f} of {width:.2f} in wide, "
          f"{tight.y1 - tight.y0 + 2 * pad:.2f} in tall")


def hide_y_grid_behind(ax):
    """Recessive horizontal grid, drawn behind the marks."""
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
