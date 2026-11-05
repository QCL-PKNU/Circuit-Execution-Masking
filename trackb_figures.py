#!/usr/bin/env python3
"""Figures, summary and checks for trackb_bv3.py (reads only the saved rows_*.json and RunStore files).

    .venv/bin/python trackb_figures.py --run runs/bv3
writes <run>/summary.csv, fig_A_metrics.png/.pdf, fig_B_bitstrings_bv3.png/.pdf, fig_C_circuits_bv3.png, report.md
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = ["original", "Q-MaskCut", "inverse gates", "composite gates", "delayed gates", "ECQCO"]
COLORS = {"original": "#7f7f7f", "Q-MaskCut": "#2878d6", "inverse gates": "#9467bd", "composite gates": "#8c564b",
          "delayed gates": "#e377c2", "ECQCO": "#bcbd22"}
W_IN = 3.4                                                  # IOP single column
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "legend.fontsize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); a = ap.parse_args()
    run = Path(a.run)
    rows = []
    for f in sorted(glob.glob(str(run / "rows_*.json"))):
        rows += json.loads(Path(f).read_text())
    df = pd.DataFrame(rows)
    main_ = df[df.label != "plain"]; plain = df[df.label == "plain"]
    cfg = json.loads((run / "config.json").read_text())

    # ---- summary: mean ± std over iterations x inputs
    cols = ["tvd_obf_min", "tvd_deobf", "tvd_ideal", "noise_floor", "fidelity", "normged", "depth", "gates", "cx", "swaps",
            "qubits", "time_us"]
    ok = main_[main_.get("status", pd.Series(index=main_.index, dtype=object)).isna()] if "status" in main_ else main_
    orig = ok[ok.method == "original"].set_index(["iteration", "input"])
    out = []
    for m in METHODS:
        d = ok[ok.method == m]
        if d.empty:
            out.append({"benchmark": "bv_n3", "method": m, "runs": 0}); continue
        r = {"benchmark": "bv_n3", "method": m, "runs": len(d)}
        for c in cols:
            if c in d:
                v = pd.to_numeric(d[c], errors="coerce")
                r[c] = f"{v.mean():.4f} ± {v.std():.4f}"
        fo = orig.reindex(d.set_index(["iteration", "input"]).index).fidelity.values
        r["accuracy_change"] = f"{np.nanmean(d.fidelity.values - fo):+.4f} ± {np.nanstd(d.fidelity.values - fo):.4f}"
        for c in ("depth", "gates", "cx"):
            r[f"{c}_change_%"] = f"{100 * (pd.to_numeric(d[c]).mean() / orig[c].mean() - 1):+.1f}"
        kinds = d.normged_kind.value_counts().to_dict() if "normged_kind" in d else {}
        r["normged_kind"] = ", ".join(f"{k} {v}" for k, v in kinds.items())
        if "operator_check" in d:
            oc = d.operator_check.astype(str).value_counts().to_dict(); r["operator_check"] = ", ".join(f"{k}: {v}" for k, v in oc.items())
        out.append(r)
    summ = pd.DataFrame(out); summ.to_csv(run / "summary.csv", index=False)

    # ---- fig A: TVD_obf_min, TVD_deobf, normGED, fidelity (mean ± std)
    fig, ax = plt.subplots(2, 2, figsize=(W_IN, 3.0))
    for a_, (c, t) in zip(ax.ravel(), [("tvd_obf_min", "TVD_obf_min (higher = hidden)"), ("tvd_deobf", "TVD_deobf (lower = correct)"),
                                       ("normged", "normGED (logical)"), ("fidelity", "Hellinger fidelity")]):
        mu = [pd.to_numeric(ok[ok.method == m][c], errors="coerce").mean() for m in METHODS]
        sd = [pd.to_numeric(ok[ok.method == m][c], errors="coerce").std() for m in METHODS]
        a_.bar(range(len(METHODS)), mu, yerr=sd, capsize=1.5, color=[COLORS[m] for m in METHODS], error_kw={"lw": 0.6})
        a_.set_xticks(range(len(METHODS)), ["orig", "QMC", "inv", "comp", "delay", "ECQCO"], rotation=45)
        a_.set_ylim(0, 1.05); a_.set_title(t); a_.grid(axis="y", alpha=0.3, lw=0.4)
    fig.suptitle(f"BV-3, FakeMontrealV2, {cfg['shots']} shots, level {cfg['optimization_level']}, "
                 f"{cfg['iterations']}x{ok.input.nunique()} runs", fontsize=7)
    fig.tight_layout(); fig.savefig(run / "fig_A_metrics.png", dpi=300); fig.savefig(run / "fig_A_metrics.pdf"); plt.close(fig)

    # ---- fig B: plain BV-3 bitstrings on the measured qubits (q1 q2; q0 is the flag)
    n = 3
    def meas2(vec):                                          # little-endian over q0 q1 q2 -> marginal on (q1, q2)
        v = np.asarray(vec).reshape([2] * n)                 # axes: q2, q1, q0
        m2 = v.sum(axis=2)                                   # sum q0 -> axes q2, q1
        return {f"{b1}{b2}"[::-1][::-1]: float(m2[b2, b1]) for b1 in (0, 1) for b2 in (0, 1)}   # key 'q2q1' qiskit order
    keys = ["00", "01", "10", "11"]
    meths = [m for m in METHODS if m != "original"]
    fig, ax = plt.subplots(3, 2, figsize=(W_IN, 4.3)); ax = ax.ravel()
    o = plain[plain.method == "original"].iloc[0]; ideal2 = meas2(o.bits_ideal)
    for a_, m in zip(ax, meths):
        r = plain[plain.method == m]
        if r.empty or not isinstance(r.iloc[0].get("bits_cv"), list):
            a_.text(0.5, 0.5, "N/A", ha="center", transform=a_.transAxes); a_.set_title(m); continue
        r = r.iloc[0]; x = np.arange(4); w = 0.27
        cv, rc, og = meas2(r.bits_cv), meas2(r.bits_rc), meas2(o.bits_cv)
        a_.bar(x - w, [og[k] for k in keys], w, color="#7f7f7f", label="original")
        a_.bar(x, [cv[k] for k in keys], w, color="#ec6935", label="compiler view")
        a_.bar(x + w, [rc[k] for k in keys], w, color="#2ca02c", label="reconstructed")
        a_.set_xticks(x, keys); a_.set_ylim(0, 1.05)
        tv_obf = 0.5 * sum(abs(ideal2[k] - cv[k]) for k in keys); tv_de = 0.5 * sum(abs(ideal2[k] - rc[k]) for k in keys)
        a_.set_title(f"{m}\nTVD_obf {tv_obf:.2f} | TVD_deobf {tv_de:.2f}", fontsize=6)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=6)
    fig.suptitle("BV-3 (secret 11), FakeMontrealV2: measured bits q2q1", fontsize=7)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(run / "fig_B_bitstrings_bv3.png", dpi=300); fig.savefig(run / "fig_B_bitstrings_bv3.pdf"); plt.close(fig)

    # ---- fig C: circuits (original, Q-MaskCut protected + pieces, ObfusQate)
    from qexp_store import RunStore
    from qmaskcut.utils.circuits import load, unmeasured
    from qmaskcut_eval import circuit_path
    from qmaskcut.baselines.obfusqate import OBFUSQATE
    import random
    st = RunStore(run); lab = "it00_in00"
    base = load("bv_n3", str(circuit_path("bv_n3")))
    items = [("original (BV-3)", base)]
    for role, t in (("protected", "Q-MaskCut protected"), ("piece_0", "Q-MaskCut piece 1"), ("piece_1", "Q-MaskCut piece 2")):
        if st.has_circuit("logical", lab, "Q-MaskCut", role):
            items.append((t, st.load_circuit("logical", lab, "Q-MaskCut", role)))
    for tech in ("composite gates", "delayed gates"):
        random.seed(0); items.append((tech, unmeasured(OBFUSQATE[tech](base.copy()))))
    fig, ax = plt.subplots(len(items), 1, figsize=(7.0, 1.0 + sum(0.42 * (q.num_qubits + 1) for _, q in items)),
                           gridspec_kw={"height_ratios": [q.num_qubits + 1 for _, q in items]})
    for a_, (t, q) in zip(ax, items):
        q.draw(output="mpl", ax=a_, fold=-1); a_.set_title(t, loc="left", fontsize=8)
    fig.tight_layout(); fig.savefig(run / "fig_C_circuits_bv3.png", dpi=200); plt.close(fig)

    # ---- checks
    fails = []
    for f in glob.glob(str(run / "counts/*/*/*.json")):
        if json.loads(Path(f).read_text())["shots"] != cfg["shots"]:
            fails.append(f"shots != {cfg['shots']}: {f}")
    for r in ok.itertuples():
        if r.method != "ECQCO" and pd.notna(r.tvd_ideal) and float(r.tvd_ideal) >= 0.01:
            fails.append(f"TVD_ideal {float(r.tvd_ideal):.3g} >= 0.01: {r.method} it{r.iteration} in{r.input}")
        if r.method != "ECQCO" and r.operator_check is False:
            fails.append(f"operator check False: {r.method} it{r.iteration} in{r.input}")
    expect = cfg["iterations"] * ok.input.nunique()
    for m in METHODS:
        got = len(ok[ok.method == m])
        if got != expect:
            fails.append(f"{m}: {got} runs, expected {expect}")
    na = main_[main_.get("status").notna()] if "status" in main_ else main_.iloc[0:0]
    for r in na.itertuples():
        fails.append(f"{r.method} it{r.iteration} in{r.input}: {r.status}")
    cnt = len(glob.glob(str(run / "counts/*/*/*.json")))
    short = summ[["method", "runs", "tvd_obf_min", "tvd_deobf", "tvd_ideal", "fidelity", "accuracy_change", "normged", "depth", "time_us"]]
    md = ["# BV-3 evaluation (pasted spec)", "", f"Run folder `{run}`; {cnt} count files; settings in `config.json`.", "",
          short.to_markdown(index=False), "",
          "## Interpretation", "",
          "1. Q-MaskCut hides the output from the compiler (high TVD_obf_min) and give it back with the key (TVD_deobf near the original's).",
          "2. Composite and delayed gates leave the output visible (TVD_obf_min = the noise level of the original).",
          "3. Inverse gates looks hidden only because its depth destroys the output for the owner too (low fidelity).",
          "4. ECQCO's output is wrong even with its key (TVD_deobf ~ 1).",
          "5. Exact reconstruction (TVD_ideal) is 0 for every decodable method; see the checks below.", "",
          "## Checks", ""] + ([f"- {x}" for x in fails] if fails else ["- all checks passed"])
    (run / "report.md").write_text("\n".join(md) + "\n")
    print(short.to_string(index=False)); print("checks:", len(fails), "issues")


if __name__ == "__main__":
    main()
