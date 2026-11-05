"""New-evaluation figures in the manuscript house style (qmaskcut_style.py), read only from the saved runs:
  runs/tier1_tier2      (Tier 1: 11 circuits, noiseless / FakeMontrealV2 / FakeMarrakesh; Tier 2: structure, 3-140 qubits)
  runs/tier1_uniformkey (Tier 1, Q-MaskCut only, uniform random key: replaces V4's Q-MaskCut rows)
  runs/bv3 (BV-3, 20 iterations x 10 random U3 inputs, FakeMontrealV2, 1024 shots, level 3, uniform key)
Each figure: paper_figs/<fig>/data/<fig>.csv (the exact numbers drawn) and paper_figs/<fig>/out/<fig>.pdf + 600-dpi png.

    .venv/bin/python paper_figs/make_v4.py
"""
import glob, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qmaskcut_style import (house_figure, house_rows, grouped_bars, finish, finish_rows, save_fig, check_overlaps,
                            LABEL_FS, PRINT_W, FIG_H, apply_style, ygrid, scheme_bar_kw, SCHEME_COLOR, S as HS)

V4 = f"{ROOT}/runs/tier1_tier2/metrics"
# Q-MaskCut with a uniform random key: its Tier-1 rows come from this run and replace the
# best-TVD rows of V4; the original and the baselines do not depend on the key and stay those of V4. Tier 2 already
# used a random key.
V4_UNIFORM = f"{ROOT}/runs/tier1_uniformkey/metrics"
BV3 = f"{ROOT}/runs/bv3"     # level 3, uniform random key
KEY = {"original": "Original", "Q-MaskCut": "Q-MaskCut", "inverse gates": "Inverse gates", "composite gates": "Composite gates",
       "delayed gates": "Delayed gates", "ECQCO": "ECQCO"}
ORDER = ["Original", "ECQCO", "Inverse gates", "Delayed gates", "Composite gates", "Q-MaskCut"]
# normGED / depth / execution-time figures: the 23 circuits of the ibm_marrakesh hardware run (same order, same names)
HW23 = ["bv_n3", "bv_n4", "bell_n4", "qpe_n9", "adder_n10", "seca_n11", "multiply_n13", "bv_n14", "qf21_n15", "qec9xz_n17",
        "bv_n19", "cat_state_n22", "ghz_state_n23", "knn_n25", "ising_n26", "ising_n34", "ghz_n40", "ising_n42", "dj_n50",
        "cat_n65", "dj_n70", "ghz_n78", "ghz_n127"]                 # = scripts/hw_final_v4.py CIRCS
SHOW = {"dj_n50": "dj_n50 (MQT)", "dj_n70": "dj_n70 (MQT)"}
T1 = ["bv_n3", "bv_n4", "bell_n4", "qpe_n9", "adder_n10", "seca_n11", "multiply_n13", "bv_n14", "qf21_n15", "qec9xz_n17", "bv_n19"]


def load_rows(backend):
    rows = []
    new = glob.glob(f"{V4_UNIFORM}/{backend}/*.json")
    for f in glob.glob(f"{V4}/{backend}/*.json"):
        rows += [r for r in json.load(open(f)) if not (new and r.get("method") == "Q-MaskCut")]
    for f in new:
        rows += json.load(open(f))
    d = pd.DataFrame(rows); d["scheme"] = d.method.map(KEY)
    return d[d.scheme.notna()]


def na_text(status):
    s = str(status)
    if s.startswith("not simulated"):
        return "too large to simulate"
    if s.startswith("not computed"):
        return "not computed"
    return "not supported"


def table(d, circuits, col):
    """wide table circuit x scheme with values, plus a parallel table of NA labels."""
    val = {k: [] for k in ORDER}; lab = {k: [] for k in ORDER}; n = []
    for c in circuits:
        dc = d[d.circuit == c]
        n.append(int(dc.n.dropna().iloc[0]) if dc.n.notna().any() else int(c.rsplit("_n", 1)[1]))
        for k in ORDER:
            r = dc[dc.scheme == k]
            ok = r[r.status == "ok"] if "status" in r else r
            val[k].append(float(ok[col].iloc[0]) if len(ok) and pd.notna(ok[col].iloc[0]) else np.nan)
            lab[k].append(na_text(r.status.iloc[0]) if len(r) and not len(ok) else "not supported")
    return n, val, lab


def save_data(fig_name, circuits, n, val, extra=None):
    os.makedirs(f"{HERE}/{fig_name}/data", exist_ok=True)
    df = pd.DataFrame({"circuit": circuits, "n": n, **val})
    if extra:
        for k, v in extra.items():
            df[k] = v
    df.to_csv(f"{HERE}/{fig_name}/data/{fig_name}.csv", index=False)


def bar_fig(fig_name, d, circuits, col, ylabel, ylim, yticks, fmt, title, schemes=ORDER, log=False, floor=None, rows=1, only=None):
    n, val, lab = table(d, circuits, col)
    save_data(fig_name, circuits, n, {k: val[k] for k in schemes})
    na = lambda name, i: lab[name][i]                                                    # noqa: E731
    if rows == 1:
        fig, ax, s = house_figure(4.0 if len(circuits) > 12 else FIG_H)
        x = grouped_bars(ax, s, schemes, {k: val[k] for k in schemes}, log=log, fmt=fmt, floor=floor, label_fs=LABEL_FS, na=na)
        finish(fig, ax, s, x, [f"{SHOW.get(c, c)} ({k})" for c, k in zip(circuits, n)], ylabel, ylim, yticks, ncol=4, title=title)
        fig.tight_layout(rect=[0, 0.13, 1, 1], pad=0.3)            # two-line legend below the axes
    else:
        h = -(-len(circuits) // rows); parts = [list(range(i, min(i + h, len(circuits)))) for i in range(0, len(circuits), h)]
        fig, axes, s = house_rows(len(parts), 2.5 if rows == 2 else 2.2)
        xs = []
        for ax, idx in zip(axes, parts):
            nap = (lambda idx_: (lambda name, i: lab[name][idx_[i]]))(idx)
            xs.append(grouped_bars(ax, s, schemes, {k: [val[k][i] for i in idx] for k in schemes}, log=log, fmt=fmt,
                                   floor=floor, label_fs=LABEL_FS, na=nap, only=only))
        finish_rows(fig, axes, s, xs, [[f"{SHOW.get(circuits[i], circuits[i])} ({n[i]})" for i in idx] for idx in parts], ylabel, ylim, yticks,
                    ncol=4)
        fig.tight_layout(rect=[0, 0, 1, 0.92], pad=0.3, h_pad=0.6)  # two-line legend on top
        # two-row figures: no title (the legend sits on top), as the other two-row manuscript figures
    save_fig(fig, f"{HERE}/{fig_name}/out", fig_name)
    check_overlaps(fig); plt.close(fig)


def tvd_pair(backend, nm):
    """TVD without the key (obfuscated, what the provider sees) above TVD with the key (deobfuscated), same circuits."""
    d = load_rows(backend); circuits = T1
    n, vo, lo = table(d, circuits, "tvd_obf"); _, vd, ld = table(d, circuits, "tvd_deobf")
    save_data(f"fig_v4_tvd_{nm}", circuits, n, {f"{k} obf": vo[k] for k in ORDER}, {f"{k} deobf": vd[k] for k in ORDER})
    fig, axes, s = house_rows(2, 2.4)
    xs = [grouped_bars(ax, s, ORDER, v, fmt="{:.2f}", label_fs=LABEL_FS, na=(lambda lab_: (lambda name, i: lab_[name][i]))(lab))
          for ax, v, lab in zip(axes, (vo, vd), (lo, ld))]
    tl = [f"{c} ({k})" for c, k in zip(circuits, n)]
    finish_rows(fig, axes, s, xs, [tl, tl], "TVD", (0, 1.3), np.arange(0, 1.01, 0.2), ncol=4)
    axes[0].set_ylabel("TVD$_{\\mathrm{obf}}$"); axes[1].set_ylabel("TVD$_{\\mathrm{deobf}}$")
    axes[0].set_xlabel("")
    fig.tight_layout(rect=[0, 0, 1, 0.92], pad=0.3, h_pad=0.6)
    save_fig(fig, f"{HERE}/fig_v4_tvd_{nm}/out", f"fig_v4_tvd_{nm}")
    check_overlaps(fig); plt.close(fig)


def fid_pair():
    """Fidelity of the decoded output on the two noise models (fair run), FakeMarrakesh above FakeMontrealV2."""
    circuits = T1; panels = []
    for b in ("FakeMarrakesh", "FakeMontrealV2"):
        n, v, lab = table(load_rows(b), circuits, "fidelity"); panels.append((v, lab))
    save_data("fig_v4_fidelity_noisy", circuits, n, {f"{k} Marrakesh": panels[0][0][k] for k in ORDER},
              {f"{k} Montreal": panels[1][0][k] for k in ORDER})
    fig, axes, s = house_rows(2, 2.4)
    xs = [grouped_bars(ax, s, ORDER, v, fmt="{:.2f}", label_fs=LABEL_FS, na=(lambda lab_: (lambda name, i: lab_[name][i]))(lab))
          for ax, (v, lab) in zip(axes, panels)]
    tl = [f"{c} ({k})" for c, k in zip(circuits, n)]
    finish_rows(fig, axes, s, xs, [tl, tl], "fidelity", (0, 1.3), np.arange(0, 1.01, 0.2), ncol=4)
    axes[0].set_ylabel("fidelity, FakeMarrakesh"); axes[1].set_ylabel("fidelity, FakeMontrealV2"); axes[0].set_xlabel("")
    fig.tight_layout(rect=[0, 0, 1, 0.92], pad=0.3, h_pad=0.6)
    save_fig(fig, f"{HERE}/fig_v4_fidelity_noisy/out", "fig_v4_fidelity_noisy")
    check_overlaps(fig); plt.close(fig)


def hw_fig():
    """ibm_marrakesh, full method, 2048 shots: fidelity of the original (compiled at level 3, run as is) and of
    Q-MaskCut rebuilt and decoded with the key. results/hw_final_v4/ibm_marrakesh/results.csv"""
    d = pd.read_csv(f"{ROOT}/results/hw_final_v4/ibm_marrakesh/results.csv")
    d["order"] = d.circuit.map({c: i for i, c in enumerate(HW23)}); d = d.sort_values("order")
    val = {"Original": d.fid_original.tolist(), "Q-MaskCut": d.fid_qmaskcut.tolist()}
    circuits = d.circuit.tolist(); n = d.n.tolist()
    save_data("fig_v4_hw_fidelity", circuits, n, val)
    fig, ax, s = house_figure(4.0)
    x = grouped_bars(ax, s, list(val), val, fmt="{:.2f}", label_fs=LABEL_FS)
    finish(fig, ax, s, x, [f"{SHOW.get(c, c)} ({k})" for c, k in zip(circuits, n)], "fidelity", (0, 1.2),
           np.arange(0, 1.01, 0.2), ncol=2)
    fig.tight_layout(rect=[0, 0.08, 1, 1], pad=0.3)
    save_fig(fig, f"{HERE}/fig_v4_hw_fidelity/out", "fig_v4_hw_fidelity"); check_overlaps(fig); plt.close(fig)

def bv3_spec():
    rows = []
    for f in sorted(glob.glob(f"{BV3}/rows_*.json")):
        rows += json.load(open(f))
    d = pd.DataFrame(rows); d["scheme"] = d.method.map(KEY)
    # normGED on the COMPILED circuits (as every other normGED in the paper; Q-MaskCut's twirl + DD are added after
    # compilation, so the logical normGED misses them): trackb_compiled_normged.py -> normged_compiled.json
    comp = {r["label"]: r for r in json.load(open(f"{BV3}/normged_compiled.json"))}
    d["normged_logical"] = d["normged"]
    d["normged"] = [0.0 if mth == "original" else (comp.get(lbl, {}).get(mth, [np.nan])[0]) for lbl, mth in zip(d.label, d.method)]
    m = d[d.label != "plain"]
    metrics = [("tvd_obf_min", "TVD$_{obf}$ min"), ("tvd_deobf", "TVD$_{deobf}$"), ("fidelity", "fidelity"), ("normged", "normGED (compiled)")]
    mu = {k: [m[m.scheme == k][c].astype(float).mean() for c, _ in metrics] for k in ORDER}
    sd = {k: [m[m.scheme == k][c].astype(float).std() for c, _ in metrics] for k in ORDER}
    os.makedirs(f"{HERE}/fig_v4_bv3_metrics/data", exist_ok=True)
    pd.DataFrame([{"scheme": k, **{f"{c}_mean": mu[k][i] for i, (c, _) in enumerate(metrics)},
                   **{f"{c}_std": sd[k][i] for i, (c, _) in enumerate(metrics)},
                   "normged_logical_mean": m[m.scheme == k].normged_logical.astype(float).mean(), "runs": int((m.scheme == k).sum())} for k in ORDER]
                 ).to_csv(f"{HERE}/fig_v4_bv3_metrics/data/fig_v4_bv3_metrics.csv", index=False)
    fig, ax, s = house_figure()
    x = grouped_bars(ax, s, ORDER, mu, fmt="{:.2f}", label_fs=LABEL_FS, only=[])      # bars only; labels below
    k = len(ORDER); w = 0.88 / k
    for j, name in enumerate(ORDER):                                                   # error bars = std
        xs = x + (j - (k - 1) / 2) * w
        ax.errorbar(xs, mu[name], yerr=sd[name], fmt="none", ecolor="black", elinewidth=0.6, capsize=1.2)
        for xi, m_, s_ in zip(xs, mu[name], sd[name]):                                 # value ABOVE the error bar
            if m_ == m_:
                ax.text(xi, m_ + (s_ if s_ == s_ else 0) + 0.02, f"{m_:.2f}", ha="center", va="bottom", rotation=90,
                        fontsize=LABEL_FS * s, fontweight="bold" if name == "Q-MaskCut" else None)
    finish(fig, ax, s, x, [t for _, t in metrics], "value (mean $\\pm$ std)", (0, 1.45), np.arange(0, 1.01, 0.2), ncol=4,
           xlabel="metric", rotation=0, title="BV-3, FakeMontrealV2, level 3, 20 iterations $\\times$ 10 random inputs")
    fig.tight_layout(rect=[0, 0.13, 1, 1], pad=0.3)
    save_fig(fig, f"{HERE}/fig_v4_bv3_metrics/out", "fig_v4_bv3_metrics"); check_overlaps(fig); plt.close(fig)

    # bitstrings of the plain BV-3, measured qubits q2 q1
    p = d[d.label == "plain"]; o = p[p.method == "original"].iloc[0]
    keys = [format(i, "03b") for i in range(8)]                                          # all 8 outcomes, 'q2q1q0' (q0 = flag)
    def m2(vec):
        v = np.asarray(vec, dtype=float)                                                 # little-endian: index bit j = qubit j
        return [v[int(k_, 2)] for k_ in keys]
    ideal = m2(o.bits_ideal)
    sch = [k for k in ORDER if k != "Original"]
    apply_style(HS)
    plt.rcParams.update({"axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7.5, "font.size": 8,
                         "axes.linewidth": 0.8, "hatch.linewidth": 0.5, "savefig.bbox": "standard"})
    fig, axes = plt.subplots(2, 3, figsize=(PRINT_W, 3.8), sharey=True); fig._fixed_size = True
    recs = []
    for a, k in zip(axes.ravel(), sch):
        r = p[p.scheme == k].iloc[0]; cv, rc, og = m2(r.bits_cv), m2(r.bits_rc), m2(o.bits_cv)
        x = np.arange(8); w = 0.27
        a.bar(x - w, og, w, color=SCHEME_COLOR["Original"], edgecolor="black", lw=0.4, label="original")
        a.bar(x, cv, w, color="#E08A2E", edgecolor="black", lw=0.4, hatch="//", label="obfuscated")
        a.bar(x + w, rc, w, color=SCHEME_COLOR[k] if k != "Composite gates" else "#FFF5E4", edgecolor="black", lw=0.4,
              hatch="..", label="deobfuscated")
        to = 0.5 * sum(abs(i - c) for i, c in zip(ideal, cv)); td = 0.5 * sum(abs(i - c) for i, c in zip(ideal, rc))
        a.set_title(f"{k}\nTVD$_{{obf}}$ {to:.2f}  TVD$_{{deobf}}$ {td:.2f}", fontsize=7.5)
        a.set_xticks(x, keys, rotation=90, fontsize=6.5); a.set_ylim(0, 1.05); ygrid(a, HS)
        recs.append({"scheme": k, **{f"obf_{kk}": v for kk, v in zip(keys, cv)}, **{f"deobf_{kk}": v for kk, v in zip(keys, rc)},
                     "tvd_obf": to, "tvd_deobf": td})
    for a in axes[:, 0]:
        a.set_ylabel("probability")
    # the free panel(s) hold the full legend: bar types + every method's colour (reconstructed bars use it)
    from matplotlib.patches import Patch
    free = list(axes.ravel())[len(sch):]
    for a in free:
        a.axis("off")
    h = [Patch(facecolor=SCHEME_COLOR["Original"], edgecolor="black", lw=0.4, label="original (correct output)"),
         Patch(facecolor="#E08A2E", edgecolor="black", lw=0.4, hatch="//", label="obfuscated")]
    h += [Patch(facecolor=SCHEME_COLOR[k] if k != "Composite gates" else "#FFF5E4", edgecolor="black", lw=0.4, hatch="..",
                label=f"deobfuscated: {k}") for k in sch]
    if free:
        free[0].legend(handles=h, loc="center", frameon=False, fontsize=7, handlelength=1.6, labelspacing=0.45)
    else:
        fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=[0.02, 0, 1, 1 if free else 0.93], pad=0.4, h_pad=0.8, w_pad=0.6)
    os.makedirs(f"{HERE}/fig_v4_bv3_bitstrings/data", exist_ok=True)
    pd.DataFrame(recs).to_csv(f"{HERE}/fig_v4_bv3_bitstrings/data/fig_v4_bv3_bitstrings.csv", index=False)
    save_fig(fig, f"{HERE}/fig_v4_bv3_bitstrings/out", "fig_v4_bv3_bitstrings"); check_overlaps(fig); plt.close(fig)


def main():
    for b, nm in (("FakeMontrealV2", "montreal"), ("FakeMarrakesh", "marrakesh"), ("none", "noiseless")):
        d = load_rows(b)
        bar_fig(f"fig_v4_fidelity_{nm}", d, T1, "fidelity", "fidelity", (0, 1.25), np.arange(0, 1.01, 0.2), "{:.2f}",
                f"Fidelity after decoding, {'noiseless' if b == 'none' else b}")
        bar_fig(f"fig_v4_tvd_obf_{nm}", d, T1, "tvd_obf", "TVD without key", (0, 1.25), np.arange(0, 1.01, 0.2), "{:.2f}",
                f"Output hiding (what the cloud sees), {'noiseless' if b == 'none' else b}")
    t2 = load_rows("structure_FakeMarrakesh")
    lab = ["Original", "Q-MaskCut"]                                        # value labels: original and ours only
    bar_fig("fig_v4_normged_all", t2, HW23, "normged", "normGED", (0, 1.3), np.arange(0, 1.01, 0.2), "{:.2f}",
            "", rows=1)
    bar_fig("fig_v4_time_all", t2, HW23, "time_us", r"execution time ($\mu$s, log)", (1, 5000), None, "{:.0f}",
            "", log=True, floor=1, rows=1)
    bar_fig("fig_v4_depth_all", t2, HW23, "depth", "compiled depth (layers, log)", (1, 400000), None, "{:.0f}",
            "", log=True, floor=1, rows=1)
    tvd_pair("none", "noiseless"); tvd_pair("FakeMarrakesh", "marrakesh"); tvd_pair("FakeMontrealV2", "montreal"); fid_pair(); hw_fig()
    bv3_spec()
    print("v4 figures built")


if __name__ == "__main__":
    main()
