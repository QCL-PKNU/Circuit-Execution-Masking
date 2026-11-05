"""qmaskcut_style - shared plotting style for the Q-MaskCut manuscript figures.

Follows the style of SUTURE's paper_figs/rasengan_style.py (from Rasengan's reproduce notebooks):
  - Times New Roman (Liberation Serif where Times is not installed), pdf/ps fonttype 42, font 60 on a
    22x12-inch canvas, scaled proportionally per figure
  - axes.linewidth 5, thick lines, black-edged bars and markers
  - dashed y-grid (linestyle '--', axis='y')
  - palette: navy #384B70, green #6A9C89, red #B8001F, cream #FFF5E4, orange #E08A2E, gray #8d97a9;
    our method in red, as SUTURE's "(ours)" arm; every scheme also has its own hatch (grayscale safe)
  - legends frameless, top-anchored, multi-column; text-overlap check after every figure

Usage:
    from qmaskcut_style import figure, ygrid, scheme_bar_kw, save_fig, check_overlaps, top_legend
    fig, ax, s = figure(scale=0.5, wratio=2.4)
    ...
    save_fig(fig, OUT, "fig_normged")      # fig_normged.pdf + fig_normged_600dpi.png
"""
import os

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

PAL = {
    "navy":   "#384B70",
    "green":  "#6A9C89",
    "red":    "#B8001F",
    "cream":  "#FFF5E4",
    "orange": "#E08A2E",
    "gray":   "#8d97a9",
    "greent": "#a9bfb7",
    "redt":   "#c6767f",
}
# fixed colour and hatch per scheme, the same in every figure
SCHEME_COLOR = {"Original": PAL["gray"], "ECQCO": PAL["navy"], "Inverse gates": PAL["orange"],
                "Delayed gates": PAL["green"], "Composite gates": PAL["cream"], "Q-MaskCut": PAL["red"]}
SCHEME_HATCH = {"Original": "", "ECQCO": "\\\\", "Inverse gates": "xx", "Delayed gates": "..",
                "Composite gates": "//", "Q-MaskCut": "/"}
SCHEME_LABEL = {"Q-MaskCut": "Q-MaskCut"}   # no "(ours)": the paper evaluates it as one example
LS = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]
MK = ["o", "s", "^", "D", "v", "P", "X"]

BASE = dict(figsize=(22, 12), font=60, axlw=5, linelw=4, mks=22, mklw=3,
            grid_lw=1.5, legfs=64, tickfs=52, labelfs=56, titlefs=68)


def apply_style(scale=0.5):
    s = scale
    mpl.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "serif",
        "font.size": BASE["font"] * s,
        "axes.linewidth": BASE["axlw"] * s,
        "axes.labelsize": BASE["labelfs"] * s,
        "axes.titlesize": BASE["titlefs"] * s,
        "xtick.labelsize": BASE["tickfs"] * s,
        "ytick.labelsize": BASE["tickfs"] * s,
        "legend.fontsize": BASE["legfs"] * s,
        "lines.linewidth": BASE["linelw"] * s,
        "lines.markersize": BASE["mks"] * s,
        "lines.markeredgewidth": BASE["mklw"] * s,
        "xtick.major.width": BASE["axlw"] * s * 0.8,
        "ytick.major.width": BASE["axlw"] * s * 0.8,
        "xtick.major.size": 10 * s,
        "ytick.major.size": 10 * s,
        "hatch.linewidth": 1.2,
        "figure.dpi": 100,
        "savefig.bbox": "tight",
    })
    return s


def figure(scale=0.5, ncols=1, nrows=1, wratio=1.0, hratio=1.0):
    s = apply_style(scale)
    fig, axes = plt.subplots(nrows, ncols, figsize=(22 * s * wratio, 12 * s * hratio))
    return fig, axes, s


def ygrid(ax, s=0.5):
    ax.grid(True, linestyle="--", linewidth=BASE["grid_lw"] * s * 1.4, axis="y", color="#999999", alpha=0.8)
    ax.set_axisbelow(True)


def scheme_bar_kw(name, s=0.5):
    return dict(color=SCHEME_COLOR[name], edgecolor="black", linewidth=BASE["axlw"] * s * 0.45,
                hatch=SCHEME_HATCH[name], label=SCHEME_LABEL.get(name, name))


def top_legend(fig, ncol, y=0.98):
    return fig.legend(loc="lower center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, y), borderaxespad=0,
                      columnspacing=0.6, handletextpad=0.3, labelspacing=0.25, handlelength=1.6)


def save_fig(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    pdf = os.path.join(outdir, f"{name}.pdf")
    png = os.path.join(outdir, f"{name}_600dpi.png")
    fixed = getattr(fig, "_fixed_size", False)     # house figures: exact size, so LaTeX never rescales them
    fig.savefig(pdf, bbox_inches=None if fixed else "tight")
    fig.savefig(png, dpi=600, bbox_inches=None if fixed else "tight")
    print(f"saved {pdf} + {png}")
    return pdf, png


def check_overlaps(fig):
    """Print TEXT OVERLAPS when two labels overprint (value labels with digits are ignored)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text) if t.get_text().strip() and t.get_visible()]
    ov = [(a.get_text()[:16], b.get_text()[:16]) for i, (a, ba) in enumerate(texts)
          for b, bb in texts[i + 1:] if ba.overlaps(bb)]
    ov = [o for o in ov if not any(ch.isdigit() for ch in o[0] + o[1])]
    if ov:
        print("TEXT OVERLAPS:", ov[:6])
    return ov


def grouped_bars(ax, s, schemes, values, log=False, fmt="{:.2f}", floor=None, label_fs=24, na="n/a", only=None,
                 label_rotation=90):
    """One group per x position, one black-edged bar per scheme. values[name]: list (nan = no value, labelled `na`;
    `na` is one label for every bar, or a callable na(name, i) giving the label of scheme `name` at position i).
    A value below `floor` is drawn at the floor (log axes) and labelled with its own value."""
    import numpy as np
    k = len(schemes); w = 0.88 / k; x = np.arange(len(values[schemes[0]]))
    for j, name in enumerate(schemes):
        y = np.asarray(values[name], dtype=float); xs = x + (j - (k - 1) / 2) * w
        shown = np.where(np.isnan(y), 0.0, y if floor is None else np.maximum(y, floor))
        ax.bar(xs, shown, w, **scheme_bar_kw(name, s))
        for i, (xi, yi, hi) in enumerate(zip(xs, y, shown)):
            if only is not None and name not in only and not np.isnan(yi):
                continue                       # dense figures: value labels for the listed schemes only
            base = (floor if floor is not None else 0) if np.isnan(yi) else hi
            lab = na(name, i) if callable(na) else na
            ax.text(xi, base * 1.08 if log else base + 0.012, lab if np.isnan(yi) else (fmt(yi) if callable(fmt) else fmt.format(yi)),
                    ha="center", va="bottom", rotation=label_rotation, fontsize=label_fs * s,
                    fontweight="bold" if name == "Q-MaskCut" else None)
    if log:
        ax.set_yscale("log")
    return x


# ---- one layout for every figure, drawn at its printed size ------------------------------------------------------
# Physica Scripta text width 153 mm; every figure is 95 % of it and is placed at that width, so the font sizes below
# are the sizes on paper (no scaling down in LaTeX).
PRINT_W = 153 / 25.4              # inches: the full text width
FIG_H = 3.3                       # inches
S = 0.18                          # line widths and label sizes of the shared helpers scale with this
LABEL_FS = 5.4 / S                # value labels: 5.4 pt on paper, the same in every figure
TICK_FS = 6.5                     # circuit names, pt
ROTATION = 40
GROUP_FACTOR = 0.9                # < 1 widens the gap between circuit groups
NA_LABEL = "not supported"
XLABEL = "benchmark circuit (qubits)"


def house_figure(height=FIG_H):
    s = apply_style(S)
    mpl.rcParams.update({"axes.labelsize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
                         "font.size": 8, "hatch.linewidth": 0.5, "axes.linewidth": 0.8,
                         "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                         "savefig.bbox": "standard"})   # exact page size, no cropping
    fig, ax = plt.subplots(figsize=(PRINT_W, height))
    fig._fixed_size = True
    return fig, ax, s


def scale_groups(ax, centers, factor=GROUP_FACTOR):
    """Pull every group's bars toward the group centre (factor < 1: wider gap between groups); labels move too."""
    import numpy as np
    centers = np.asarray(centers, dtype=float)
    for cont in ax.containers:
        for rect in getattr(cont, "patches", cont):
            if not hasattr(rect, "get_x"):
                continue
            cx = rect.get_x() + rect.get_width() / 2
            gc = centers[np.abs(centers - cx).argmin()]
            w = rect.get_width() * factor
            rect.set_width(w); rect.set_x(gc + (cx - gc) * factor - w / 2)
    for t in ax.texts:
        tx, ty = t.get_position()
        gc = centers[np.abs(centers - tx).argmin()]
        t.set_position((gc + (tx - gc) * factor, ty))
    for ln in ax.lines:                                     # error bars and markers
        xs = np.asarray(ln.get_xdata(), dtype=float)
        if xs.size:
            g = centers[np.abs(centers[None, :] - xs[:, None]).argmin(axis=1)]
            ln.set_xdata(g + (xs - g) * factor)
    for coll in ax.collections:                             # error-bar line collections
        try:
            segs = coll.get_segments()
        except AttributeError:
            continue
        new = []
        for sgm in segs:
            sgm = np.array(sgm, dtype=float)
            g = centers[np.abs(centers[None, :] - sgm[:, :1]).argmin(axis=1)]
            sgm[:, 0] = g + (sgm[:, 0] - g) * factor; new.append(sgm)
        coll.set_segments(new)


def color_na_labels(ax, na_label=NA_LABEL):
    """Colour each 'not supported' label with the colour of the method whose bar slot it sits in."""
    slots = [(r.get_x() + r.get_width() / 2, r.get_facecolor()) for cont in ax.containers
             for r in getattr(cont, "patches", cont) if hasattr(r, "get_x")]
    for t in ax.texts:
        if t.get_text() == na_label and slots:
            t.set_color(min(slots, key=lambda c: abs(c[0] - t.get_position()[0]))[1])


def finish(fig, ax, s, x, ticklabels, ylabel, ylim, yticks=None, ncol=5, xlabel=XLABEL, rotation=ROTATION,
           group_factor=GROUP_FACTOR, title=None):
    """Same axes, ticks, grid and legend for every figure."""
    scale_groups(ax, x, group_factor); color_na_labels(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(ticklabels, rotation=rotation, ha="right" if rotation else "center", fontsize=TICK_FS)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_ylim(*ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
    ygrid(ax, s)
    if title:                                                  # a title on every figure, as ECQCO's Figs. 4 and 5
        ax.set_title(title, fontsize=9.5, pad=4)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, 0.0),
               borderaxespad=0.2, columnspacing=1.0, handletextpad=0.35, labelspacing=0.25, handlelength=1.8)
    fig.tight_layout(rect=[0, 0.07, 1, 1], pad=0.3)                # legend under the plot, as ECQCO
    fit_labels_to_bars(fig, ax)


def fit_labels_to_bars(fig, ax, fill=0.95):
    """A vertical value label is never wider than its bar: its font size is capped at the bar width on paper."""
    rects = [r for c in ax.containers for r in getattr(c, "patches", c) if hasattr(r, "get_width")]
    if not rects:
        return
    x0 = ax.transData.transform((0, ax.get_ylim()[0]))[0]
    bar_pt = (ax.transData.transform((rects[0].get_width(), ax.get_ylim()[0]))[0] - x0) * 72 / fig.dpi
    for t in ax.texts:
        if t.get_rotation() == 90:
            t.set_fontsize(min(t.get_fontsize(), fill * bar_pt))


def house_rows(nrows=2, height_per_row=2.6):
    """The same print-size layout with several rows of circuits (wider bars for figures with many circuits)."""
    s = apply_style(S)
    mpl.rcParams.update({"axes.labelsize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
                         "font.size": 8, "hatch.linewidth": 0.5, "axes.linewidth": 0.8,
                         "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                         "savefig.bbox": "standard"})   # exact page size, no cropping
    fig, axes = plt.subplots(nrows, 1, figsize=(PRINT_W, height_per_row * nrows + 0.35))
    fig._fixed_size = True
    return fig, axes, s


def finish_rows(fig, axes, s, xs, ticklabels, ylabel, ylim, yticks=None, ncol=5, xlabel=XLABEL, rotation=ROTATION,
                group_factor=GROUP_FACTOR):
    """finish() for a figure with several rows: every row the same axes; one legend; x label on the last row."""
    n = max(len(x) for x in xs)
    for k, (ax, x, tl) in enumerate(zip(axes, xs, ticklabels)):
        scale_groups(ax, x, group_factor); color_na_labels(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(tl, rotation=rotation, ha="right" if rotation else "center", fontsize=TICK_FS)
        ax.set_xlim(-0.6, n - 0.4)                       # rows share the bar width
        ax.set_ylabel(ylabel); ax.set_ylim(*ylim)
        if yticks is not None:
            ax.set_yticks(yticks)
        ygrid(ax, s)
    axes[-1].set_xlabel(xlabel)
    h, l = axes[0].get_legend_handles_labels()            # one legend for the whole figure
    fig.legend(h, l, loc="upper center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, 1.0), borderaxespad=0.2,
               columnspacing=0.6, handletextpad=0.3, labelspacing=0.25, handlelength=1.6)
    fig.tight_layout(rect=[0, 0, 1, 0.955], pad=0.3, h_pad=0.6)
    for ax in axes:
        fit_labels_to_bars(fig, ax)
