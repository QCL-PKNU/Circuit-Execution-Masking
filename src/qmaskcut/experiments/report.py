"""report.md and the two figures, from the per-circuit results."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

from .. import config as cfg
from ..utils.logs import log

SCHEMES = ["unprotected", "Q-MaskCut", "ECQCO", "inverse gates", "composite gates",
           "delayed gates"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def write_report(results: list, out_dir: Path):
    L = ["# Q-MaskCut experiments on " + cfg.BACKEND, "",
         f"{cfg.SHOTS} shots per circuit (every fragment instance too), transpiler level "
         f"{cfg.OPT_LEVEL} for every arm, seed {cfg.SEED}. Fidelity = Hellinger fidelity to the "
         f"ideal distribution; TVD = total variation distance to it.", ""]
    lo, hi = min(r["n"] for r in results), max(r["n"] for r in results)
    head = (f"| Range | Circuit | Qubits | Pieces ({cfg.MAX_CUTS} cuts) | Fidelity uncut | "
            f"Fidelity {cfg.MAX_CUTS} cuts | Gain | TVD uncut | TVD {cfg.MAX_CUTS} cuts | Backend | Selected |")
    L += ["## Q-MaskCut: unprotected uncut vs full Q-MaskCut", "",
          "Uncut = the original circuit run whole, unprotected. "
          f"{cfg.MAX_CUTS} cuts = the FULL Q-MaskCut method: decoys, idle padding, CZ padlocks "
          "and QOTP twirl, then the CutQC cut (Pieces are the protected fragment widths, "
          "which add up to n + decoys + K), fragments on the device, CutQC reconstruction "
          "and the client's XOR decode.", "", head, "|" + "---|" * 11]
    for r in results:
        u, q = r["schemes"].get("unprotected"), r["schemes"].get("Q-MaskCut")
        if not (u and q):
            continue
        L.append(f"| {lo}-{hi} | {r['name']} | {r['n']} | {q['pieces']} | {u['fidelity']:.3f} | "
                 f"{q['fidelity']:.3f} | {q['fidelity'] - u['fidelity']:+.3f} | {u['TVD']:.3f} | "
                 f"{q['TVD']:.3f} | {cfg.BACKEND} | {'yes' if r['selected'] else 'no (extra)'} |")
    if any(r.get("cut study", {}).get("cut") for r in results):
        L += ["", "## For reference: plain CutQC cutting, no protection", "",
              "The original circuit cut the same way but with none of the Q-MaskCut layers, "
              "so it separates what the cut does from what the protection costs.", "",
              head, "|" + "---|" * 11]
        for r in results:
            cs = r.get("cut study")
            if not cs or "cut" not in cs:
                continue
            u, c = cs["uncut"], cs["cut"]
            L.append(f"| {lo}-{hi} | {r['name']} | {r['n']} | {c['pieces']} | {u['fidelity']:.3f} | "
                     f"{c['fidelity']:.3f} | {cs['gain']:+.3f} | {u['TVD']:.3f} | {c['TVD']:.3f} | "
                     f"{cfg.BACKEND} | {'yes' if r['selected'] else 'no (extra)'} |")
    L += ["", "## Schemes", "",
          "| Circuit | scheme | qubits | gates | depth | exec depth | normGED | GED_2Q | "
          "semantics | fidelity | TVD | backend |", "|" + "---|" * 12]
    for r in results:
        for s in SCHEMES:
            a = r["schemes"].get(s)
            if not a:
                continue
            g = lambda k, d=4: f"{a[k]:.{d}f}" if isinstance(a.get(k), float) else str(a.get(k, "-"))
            L.append(f"| {r['name']} | {s} | {a['qubits']} | {a['gates']} | {a['depth']} | "
                     f"{a.get('depth_exec', '-')} | {g('normGED')} | {g('GED_2Q')} | "
                     f"{g('semantics', 3)} | {g('fidelity')} | {g('TVD')} | {a.get('backend', '-')} |")
    L += ["", "Q-MaskCut cuts (of the protected circuit):", ""]
    for r in results:
        a = r["schemes"].get("Q-MaskCut")
        if a:
            L.append(f"- {r['name']}: pieces {a['pieces']}, K = {a['K']}, {a['instances']} "
                     f"instances, exact error {a['cut_exact_err']:.1e}; cloud without the key: "
                     f"fidelity {a['cloud']['fidelity']:.4f}, TVD {a['cloud']['TVD']:.4f}")
    L += ["", "![cut study](cut_study.png)", "", "![schemes](schemes.png)"]
    (out_dir / "report.md").write_text("\n".join(L) + "\n")
    for line in L[:-4]:
        log.info(line)


def _style(ax, title, ylim=None):
    ax.set_facecolor("#fcfcfb")
    ax.set_title(title, fontsize=10, color="#0b0b0b", loc="left")
    if ylim:
        ax.set_ylim(*ylim)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7"); ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", labelsize=8)
    ax.grid(axis="y", color="#e8e7e2", linewidth=0.6); ax.set_axisbelow(True)


def make_figures(results: list, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [r["name"] for r in results]
    x = np.arange(len(names))

    rows = [r for r in results if "Q-MaskCut" in r["schemes"]]
    if rows:
        series = [("uncut (unprotected)", lambda r: r["schemes"]["unprotected"]["fidelity"], 0),
                  (f"Q-MaskCut, {cfg.MAX_CUTS} cuts", lambda r: r["schemes"]["Q-MaskCut"]["fidelity"], 1)]
        if all(r.get("cut study", {}).get("cut") for r in rows):
            series.append((f"plain CutQC, {cfg.MAX_CUTS} cuts (reference)",
                           lambda r: r["cut study"]["cut"]["fidelity"], 3))
        fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(rows) + 3), 4.6), dpi=150)
        fig.patch.set_facecolor("#fcfcfb")
        xs = np.arange(len(rows))
        bw = 0.8 / len(series)
        for j, (lab, get, ci) in enumerate(series):
            ys = [get(r) for r in rows]
            bars = ax.bar(xs - 0.4 + bw * (j + 0.5), ys, bw * 0.92, color=COLORS[ci], label=lab,
                          edgecolor="#fcfcfb", linewidth=1)
            for b, y in zip(bars, ys):
                ax.text(b.get_x() + b.get_width() / 2, y, f"{y:.2f}", ha="center",
                        va="bottom", fontsize=7, color="#52514e")
        ax.set_xticks(xs, [r["name"] for r in rows], rotation=30, ha="right")
        _style(ax, f"Fidelity on {cfg.BACKEND}: unprotected uncut vs full Q-MaskCut", (0, 1.12))
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        fig.tight_layout()
        fig.savefig(out_dir / "cut_study.png", facecolor=fig.get_facecolor())
        plt.close(fig)

    present = [s for s in SCHEMES if any(s in r["schemes"] for r in results)]
    panels = [("fidelity", f"fidelity on {cfg.BACKEND}"), ("normGED", "normGED"),
              ("GED_2Q", "GED_2Q (2-qubit structure hidden)")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(max(10, 1.3 * len(names) + 3), 10),
                             dpi=150, sharex=True)
    fig.patch.set_facecolor("#fcfcfb")
    w = 0.84 / len(present)
    for ax, (key, title) in zip(axes, panels):
        for j, s in enumerate(present):
            xs = [i - 0.42 + w * (j + 0.5) for i, r in enumerate(results)
                  if r["schemes"].get(s, {}).get(key) is not None]
            ys = [r["schemes"][s][key] for r in results
                  if r["schemes"].get(s, {}).get(key) is not None]
            ax.bar(xs, ys, w * 0.92, color=COLORS[SCHEMES.index(s)], label=s,
                   edgecolor="#fcfcfb", linewidth=0.8)
        _style(ax, title, (0, 1.05))
    axes[-1].set_xticks(x, names, rotation=30, ha="right")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(present), frameon=False,
               fontsize=8.5)
    fig.suptitle(f"Q-MaskCut vs ECQCO vs ObfusQate ({cfg.BACKEND}, {cfg.SHOTS} shots); "
                 f"values in report.md", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_dir / "schemes.png", facecolor=fig.get_facecolor())
    plt.close(fig)


def report_only(run_dir: str):
    """Rebuild report.md and the figures from a finished (or partial) run's results.json."""
    d = Path(run_dir)
    log.addHandler(logging.StreamHandler(sys.stdout)); log.setLevel(logging.INFO)
    log.propagate = False
    results = json.loads((d / "results.json").read_text())
    write_report(results, d)
    if cfg.MAKE_FIGURES:
        make_figures(results, d)
    log.info(f"rebuilt      : {d / 'report.md'} and figures ({len(results)} circuits)")
