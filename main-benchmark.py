#!/usr/bin/env python3
# .venv/bin/python main-benchmark.py
#
# normGED and depth on every benchmark circuit, 4 to 140 qubits (for the slides):
# Q-MaskCut, and optionally ECQCO and ObfusQate (inverse / composite / delayed gates).
# Nothing is executed and no QPU time is used: depth is measured by compiling against a
# real IBM device's target (coupling map + basis), loaded through the saved account.
#   normGED     original vs each scheme's circuit (for Q-MaskCut: the monolith, before the
#               cut); Eq. 14 of the ECQCO paper, deterministic GED (same value on every run)
#   depth       original circuit | Q-MaskCut run whole | Q-MaskCut after the cut (largest
#               fragment), all compiled for the same device at the same level
#   cut         small circuits: the given number of fragments, at most MAX_CUTS cuts;
#               large circuits (fragments = None): cut until every fragment fits
#               MAX_FRAGMENT_WIDTH wires, so each one can be simulated exactly
# Output: results/benchmark/<time>/ run.log, report.md, normged_cut_depth.csv (Q-MaskCut),
#         schemes_normged_depth.csv (every scheme), normged_depth.png, schemes.png
import csv
import json
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from qmaskcut import config as cfg                                   # noqa: E402

# ================================ SETTINGS =====================================
# (name, QASMBench path or None for the all-ones BV, fragments or None = cut to width)
CIRCUITS = [
    ("bell_n4",       "small/bell_n4/bell_n4.qasm",               2),
    ("qpe_n9",        "small/qpe_n9/qpe_n9.qasm",                 2),
    ("adder_n10",     "small/adder_n10/adder_n10.qasm",           2),
    ("qec9xz_n17",    "medium/qec9xz_n17/qec9xz_n17.qasm",        2),
    ("bigadder_n18",  "medium/bigadder_n18/bigadder_n18.qasm",    2),
    ("bv_n19",        "medium/bv_n19/bv_n19.qasm",                3),
    ("qram_n20",      "medium/qram_n20/qram_n20.qasm",            2),
    ("cat_state_n22", "medium/cat_state_n22/cat_state_n22.qasm",  3),
    ("ghz_state_n23", "medium/ghz_state_n23/ghz_state_n23.qasm",  3),
    ("knn_n25",       "medium/knn_n25/knn_n25.qasm",              3),
    ("ising_n26",     "medium/ising_n26/ising_n26.qasm",          3),
    ("wstate_n27",    "medium/wstate_n27/wstate_n27.qasm",        2),
    # ---- 34 to 140 qubits: cut to MAX_FRAGMENT_WIDTH
    ("ising_n34",     "large/ising_n34/ising_n34.qasm",           None),
    ("cat_n35",       "large/cat_n35/cat_n35.qasm",               None),
    ("ghz_n40",       "large/ghz_n40/ghz_n40.qasm",               None),
    ("cat_n65",       "large/cat_n65/cat_n65.qasm",               None),
    # ("wstate_n76",    "large/wstate_n76/wstate_n76.qasm",         None),
    ("ghz_n78",       "large/ghz_n78/ghz_n78.qasm",               None),
    # ("wstate_n118",   "large/wstate_n118/wstate_n118.qasm",       None),
    ("ghz_n127",      "large/ghz_n127/ghz_n127.qasm",             None),
    ("cat_n130",      "large/cat_n130/cat_n130.qasm",             None),
]
ONLY = []                         # e.g. ["ghz_n40", "bv_ones_n140"]; [] = all
SKIP = ["wstate_n76", "wstate_n118"]   # too slow: CutQC fragment build on W-state circuits
CIRCUIT_TIMEOUT = 900             # seconds per circuit; a circuit that takes longer is skipped
RESUME_FROM = None                # e.g. "results/benchmark/<time>" to keep circuits done there
MAX_FRAGMENT_WIDTH = 20           # 34-40 qubits: every fragment at most this many wires
WIDE_FRAGMENT_WIDTH = 27          # 41-140 qubits: fewer cuts, each fragment still fits a
                                  # 27-qubit device and a statevector check (as the hardware run)
RUN_ECQCO = True                  # ECQCO normGED + depth, from ECQCO's own released code
ECQCO_TIMEOUT = 1800              # seconds; a circuit ECQCO cannot finish is skipped
RUN_OBFUSQATE = True              # inverse, composite and delayed gates: normGED + depth

SETTINGS = dict(
    BACKEND="FakeMarrakesh",      # offline 156-qubit copy of ibm_marrakesh: fixed calibration,
                                  # reproducible; or "ibm_marrakesh" / "ibm_fez" (live target)
    OPT_LEVEL=3,
    SEED=42,
    MAX_CUTS=2,                   # small circuits only
    QMC_DECOYS=2,
    QMC_PADLOCKS=2,               # 2 or 4: 4 raises GED_2Q, costs some depth
    GED_SWEEPS=40,
    ASCII_MAX_OPS=200,
    ASCII_FOLD=150,
)
# ===============================================================================


def main():
    from qmaskcut.device import Device
    from qmaskcut.utils.logs import banner, log, setup_logging

    out = ROOT / "results" / "benchmark" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging(out)
    t0 = time.time()
    todo = [c for c in CIRCUITS if (not ONLY or c[0] in ONLY) and c[0] not in SKIP]
    banner(f"Q-MaskCut benchmark | normGED + depth | {len(todo)} circuits | depth on "
           f"{cfg.BACKEND} (compile only, no QPU) | level {cfg.OPT_LEVEL} | "
           f"{cfg.QMC_PADLOCKS} padlocks")
    log.info(f"output       : {out.relative_to(ROOT)}/")
    dev = Device(cfg.BACKEND)
    log.info(f"device       : {cfg.BACKEND}, {dev.width} qubits")
    exec_depth = lambda qc: dev.compile(qc).depth()                  # noqa: E731

    rows, schemes = [], []
    if RESUME_FROM:
        prev = json.loads((ROOT / RESUME_FROM / "results.json").read_text())
        keep = {c[0] for c in todo}
        rows = [r for r in prev["qmaskcut"] if r["circuit"] in keep]
        schemes = [r for r in prev["schemes"] if r["circuit"] in keep]
        log.info(f"resume       : {len(rows)} circuits kept from {RESUME_FROM}")
    done = {r["circuit"] for r in rows}
    skipped = [c for c in CIRCUITS if c[0] in SKIP]
    if skipped:
        log.info(f"skip         : {', '.join(c[0] for c in skipped)} (SKIP setting)")

    def _timeout(signum, frame):
        raise TimeoutError

    signal.signal(signal.SIGALRM, _timeout)
    for k, (name, path, pieces) in enumerate(todo, 1):
        if name in done:
            continue
        banner(f"[{k}/{len(todo)}] {name}")
        signal.alarm(CIRCUIT_TIMEOUT)
        try:
            run_one(name, path, pieces, rows, schemes, dev, exec_depth, out, log, t0)
        except TimeoutError:
            # keep what finished (the original and Q-MaskCut come first) and save it, so a
            # baseline that runs past the limit never takes finished results with it
            log.info(f"skipped      : {name}: the remaining comparison schemes took longer "
                     f"than CIRCUIT_TIMEOUT = {CIRCUIT_TIMEOUT}s")
            (out / "results.json").write_text(json.dumps({"qmaskcut": rows, "schemes": schemes},
                                                         indent=1))
            write_cut_depth_csv(rows, out / "normged_cut_depth.csv")
            write_schemes_csv(schemes, out / "schemes_normged_depth.csv")
        finally:
            signal.alarm(0)

    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    write_cut_depth_csv(rows, out / "normged_cut_depth.csv")
    banner("REPORT")
    for line in write_report(rows, out):
        log.info(line)
    make_figure(rows, out)
    if len({r["scheme"] for r in schemes}) > 1:
        for line in schemes_report(schemes, out):
            log.info(line)
        schemes_figure(schemes, out)
    log.info(f"figures      : normged_depth.png, schemes.png in {out.relative_to(ROOT)}")
    log.info(f"done in {time.time() - t0:.0f}s | log {log_path.relative_to(ROOT)}")


SCHEMES = ["Q-MaskCut", "ECQCO", "inverse gates", "composite gates", "delayed gates"]
COLORS = {"Q-MaskCut": "#eb6834", "ECQCO": "#1baf7a", "inverse gates": "#eda100",
          "composite gates": "#e87ba4", "delayed gates": "#008300"}


def baselines(base, name, out, log):
    """(scheme, circuit) for every enabled comparison scheme, built from the original."""
    import random
    import subprocess

    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, qasm2
    if RUN_ECQCO:
        n = base.num_qubits
        work = out / "ecqco"
        work.mkdir(exist_ok=True)
        qr, cr = QuantumRegister(n, "q"), ClassicalRegister(n, "c")
        src = QuantumCircuit(qr, cr)
        src.compose(base, qubits=range(n), inplace=True)
        src.measure(qr, cr)
        (work / f"{name}.qasm").write_text(qasm2.dumps(src))
        (work / f"{name}.json").write_text(json.dumps(
            [{"name": name, "qasm": str(work / f"{name}.qasm"), "qubits": n}]))
        t = time.time()
        # Our wrapper run_ecqco_noverify.py calls ECQCO's UNMODIFIED QCOO + QCSO at the
        # shipped ADOA default and skips only the 2^n verification simulation, which halts
        # above ~34 qubits and does not change the circuit it writes.
        try:
            p = subprocess.run([str(cfg.ECQCO_PY),
                                str(cfg.ECQCO_WRAPPERS / "run_ecqco_noverify.py"),
                                "--list", str(work / f"{name}.json"), "--out", str(work),
                                "--seed", str(cfg.SEED)],
                               capture_output=True, text=True, timeout=ECQCO_TIMEOUT)
        except subprocess.TimeoutExpired:
            p = None
            log.info(f"ECQCO        : skipped, did not finish within {ECQCO_TIMEOUT}s")
        if p is None:
            pass
        elif (work / name).exists():
            log.info(f"ECQCO        : compiled by ECQCO's released code (shipped settings) "
                     f"in {time.time() - t:.0f}s")
            yield "ECQCO", QuantumCircuit.from_qasm_file(str(work / name))
        else:
            log.info(f"ECQCO        : skipped, its code could not process this circuit: "
                     f"{(p.stdout + p.stderr).strip()[-200:]}")
    if RUN_OBFUSQATE:
        from qmaskcut.baselines.obfusqate import OBFUSQATE
        for tech, make in OBFUSQATE.items():
            random.seed(cfg.SEED)
            try:
                yield tech, make(base.copy())
            except Exception as e:      # noqa: BLE001 - one scheme failing must not stop the rest
                log.info(f"{tech:12s} : FAILED {type(e).__name__}: {e}")


def write_schemes_csv(schemes, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["circuit", "n", "scheme", "qubits", "gates",
                                          "normGED", "depth", "depth_cut"])
        w.writeheader()
        for r in schemes:
            w.writerow({**r, "normGED": f"{r['normGED']:.4f}"})


def schemes_report(schemes, out) -> list[str]:
    names = list(dict.fromkeys(r["circuit"] for r in schemes))
    present = [s for s in SCHEMES if any(r["scheme"] == s for r in schemes)]
    get = lambda c, s: next((r for r in schemes if r["circuit"] == c and r["scheme"] == s), None)  # noqa: E731
    L = ["", "## Every scheme: normGED (original vs the scheme's circuit)", "",
         "| circuit | n | " + " | ".join(present) + " |", "|" + "---|" * (len(present) + 2)]
    for c in names:
        L.append(f"| {c} | {get(c, present[0])['n']} | " + " | ".join(
            f"{get(c, s)['normGED']:.4f}" if get(c, s) else "-" for s in present) + " |")
    L += ["", f"## Every scheme: executed depth on {cfg.BACKEND} (Q-MaskCut: uncut / cut)", "",
          "| circuit | n | original | " + " | ".join(present) + " |",
          "|" + "---|" * (len(present) + 3)]
    orig = {r["circuit"]: r["depth_orig"] for r in json.loads((out / "results.json").read_text())["qmaskcut"]}
    for c in names:
        cells = []
        for s in present:
            r = get(c, s)
            cells.append("-" if not r else f"{r['depth']} / **{r['depth_cut']}**"
                         if s == "Q-MaskCut" else f"{r['depth']}")
        L.append(f"| {c} | {get(c, present[0])['n']} | {orig.get(c, '-')} | " + " | ".join(cells) + " |")
    with open(out / "report.md", "a") as f:
        f.write("\n".join(L) + "\n")
    return L


def schemes_figure(schemes, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    names = list(dict.fromkeys(r["circuit"] for r in schemes))
    present = [s for s in SCHEMES if any(r["scheme"] == s for r in schemes)]
    get = lambda c, s: next((r for r in schemes if r["circuit"] == c and r["scheme"] == s), None)  # noqa: E731
    x, w = np.arange(len(names)), 0.84 / len(present)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(max(13, 0.75 * len(names) + 3), 9), dpi=150,
                                 sharex=True)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, key, title, logy in ((a1, "normGED", "normGED (original vs each scheme's circuit)", False),
                                 (a2, "depth", f"executed depth on {cfg.BACKEND}, log scale "
                                               f"(Q-MaskCut: after the cut)", True)):
        ax.set_facecolor("#fcfcfb")
        for j, s in enumerate(present):
            xs, ys = [], []
            for i, c in enumerate(names):
                r = get(c, s)
                if r:
                    xs.append(i - 0.42 + w * (j + 0.5))
                    ys.append(r["depth_cut"] if (key == "depth" and s == "Q-MaskCut") else r[key])
            ax.bar(xs, ys, w * 0.92, color=COLORS[s], label=s, edgecolor="#fcfcfb", linewidth=0.6)
        ax.set_title(title, fontsize=10.5, loc="left")
        if logy:
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, 1.08)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color="#e8e7e2", linewidth=0.6); ax.set_axisbelow(True)
    a2.set_xticks(x, names, rotation=40, ha="right", fontsize=8)
    h, lab = a1.get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(present), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "schemes.png", facecolor=fig.get_facecolor())
    plt.close(fig)


def run_one(name, path, pieces, rows, schemes, dev, exec_depth, out, log, t0):
    """normGED and depth for one circuit: Q-MaskCut, then every enabled baseline."""
    from qmaskcut.cutting import cutqc as C
    from qmaskcut.experiments.pipeline import check_cut
    from qmaskcut.masking import protect_general
    from qmaskcut.metrics.normged import deterministic_normged
    from qmaskcut.utils.circuits import load, n2q, unmeasured
    from qmaskcut.utils.logs import describe, draw
    t = time.time()
    base = load(name, path)
    n = base.num_qubits
    describe("original", base)
    draw(base, "original circuit")

    prot, tags, info = protect_general(base, n_decoy=cfg.QMC_DECOYS,
                                       n_padlock=cfg.QMC_PADLOCKS, reps=2, seed=cfg.SEED)
    log.info(f"Q-MaskCut    : {n} -> {prot.num_qubits} wires, {base.size()} -> "
             f"{prot.size()} gates, {n2q(base)} -> {n2q(prot)} two-qubit")

    tg = time.time()
    ng = deterministic_normged(base, prot)
    log.info(f"normGED      : {ng:.4f}  (Eq. 14, deterministic GED, {time.time() - tg:.1f}s)")

    tc = time.time()
    parts = C.independent_pieces(prot)
    target, idle = prot, []
    if len(parts) > 1:      # already split: cut its largest piece, keep the rest as they are
        target = C.piece_circuit(prot, parts[0])
        idle = [C.piece_circuit(prot, w) for w in parts[1:]]
        log.info(f"pieces       : {len(parts)} independent pieces; cutting the "
                 f"{len(parts[0])}-wire one, {len(idle)} more kept as they are "
                 f"({'+'.join(str(len(w)) for w in parts[1:])} wires)")
    if pieces is not None:
        cq, _ = C.find_balanced_cut(target, pieces, max_cuts=cfg.MAX_CUTS,
                                    allow_extra_cut=False)
        how = f"{pieces} fragments, at most {cfg.MAX_CUTS} cuts"
    else:
        cap = MAX_FRAGMENT_WIDTH if n <= 40 else WIDE_FRAGMENT_WIDTH
        cq, pieces = C.cut_to_width(target, cap)
        how = f"cut until every fragment has at most {cap} wires"
    subs = cq.cut_solution["subcircuits"]
    K = cq.cut_solution["num_cuts"]
    all_widths = C.widths(cq) + [len(w) for w in parts[1:]]
    shown = "+".join(map(str, C.widths(cq))) + (f" (+{len(idle)} idle)" if idle else "")
    log.info(f"cut          : {how}: widths {shown}, K = {K} ({time.time() - tc:.1f}s)")
    ok = all(check_cut(cq, target, dev.width).values())

    d_orig, d_whole = exec_depth(base), exec_depth(prot)
    d_cut = max([exec_depth(s) for s in subs] + [exec_depth(s) for s in idle])
    log.info(f"depth        : original {d_orig} | Q-MaskCut uncut {d_whole} | "
             f"Q-MaskCut cut {d_cut}  (executed on {cfg.BACKEND}; as built "
             f"{base.depth()} / {prot.depth()})")
    rows.append({"circuit": name, "n": n, "N": prot.num_qubits, "gates": base.size(),
                 "gates_qmc": prot.size(), "normGED": ng,
                 "pieces": shown, "fragments": len(subs) + len(idle), "K": K,
                 "max_fragment": max(all_widths), "cut_ok": ok,
                 "depth_orig": d_orig, "depth_qmc_uncut": d_whole, "depth_qmc_cut": d_cut,
                 "depth_orig_built": base.depth(), "depth_qmc_built": prot.depth(),
                 "seconds": round(time.time() - t, 1)})
    schemes.append({"circuit": name, "n": n, "scheme": "original", "qubits": n,
                    "gates": base.size(), "normGED": 0.0, "depth": d_orig, "depth_cut": None})
    schemes.append({"circuit": name, "n": n, "scheme": "Q-MaskCut", "qubits": prot.num_qubits,
                    "gates": prot.size(), "normGED": ng, "depth": d_whole, "depth_cut": d_cut})
    for scheme, obf in baselines(base, name, out, log):
        tb = time.time()
        obf = unmeasured(obf)
        ngb, db = deterministic_normged(base, obf), exec_depth(obf)
        log.info(f"{scheme:12s} : {obf.num_qubits} qubits, {obf.size()} gates | normGED "
                 f"{ngb:.4f} | depth {db} ({time.time() - tb:.1f}s)")
        schemes.append({"circuit": name, "n": n, "scheme": scheme, "qubits": obf.num_qubits,
                        "gates": obf.size(), "normGED": ngb, "depth": db, "depth_cut": None})
    (out / "results.json").write_text(json.dumps({"qmaskcut": rows, "schemes": schemes},
                                                 indent=1))
    write_cut_depth_csv(rows, out / "normged_cut_depth.csv")
    write_schemes_csv(schemes, out / "schemes_normged_depth.csv")
    log.info(f"done         : {name} in {time.time() - t:.0f}s ({time.time() - t0:.0f}s total)")



def write_cut_depth_csv(rows, path):
    """The cut-depth table layout, plus normGED (original vs monolith)."""
    cols = ["benchmark", "qubits", "normGED", "cut_K", "cut_widths", "depth_original",
            "depth_monolith_uncut", "depth_largest_fragment", "depth_reduction_pct",
            "fragment_vs_monolith", "fragment_vs_original", "below_original", "device"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            o, m, c = r["depth_orig"], r["depth_qmc_uncut"], r["depth_qmc_cut"]
            w.writerow([r["circuit"], r["n"], f"{r['normGED']:.4f}", r["K"], r["pieces"], o, m, c,
                        f"{100 * (1 - c / m):+.0f}", f"{c / m:.3f}", f"{c / o:.3f}",
                        "yes" if c < o else "no", cfg.BACKEND])


def write_report(rows, out) -> list[str]:
    L = ["# Q-MaskCut benchmark: normGED and depth", "",
         f"Depth is the executed depth after compiling for {cfg.BACKEND} (156 qubits) at "
         f"optimisation level {cfg.OPT_LEVEL}; compile only, no QPU time. 'cut' is the "
         f"largest fragment. normGED is Eq. 14 of the ECQCO paper with a deterministic GED "
         f"search. {cfg.QMC_DECOYS} decoys, {cfg.QMC_PADLOCKS} padlocks.", "",
         "| circuit | qubits | normGED | pieces | K | depth original | depth Q-MaskCut uncut | depth Q-MaskCut cut | cut vs original |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['circuit']} | {r['n']} | {r['normGED']:.4f} | {r['pieces']} | {r['K']} | "
                 f"{r['depth_orig']} | {r['depth_qmc_uncut']} | {r['depth_qmc_cut']} | "
                 f"{r['depth_qmc_cut'] / r['depth_orig']:.2f}x |")
    ng = sorted(r["normGED"] for r in rows)
    ratio = sorted(r["depth_qmc_cut"] / r["depth_orig"] for r in rows)
    med = lambda v: v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2  # noqa: E731
    L += ["", f"Median normGED {med(ng):.4f} (range {ng[0]:.4f} to {ng[-1]:.4f}); median "
              f"executed depth after the cut {med(ratio):.2f}x the original. "
              f"Cut checks: {sum(r['cut_ok'] for r in rows)}/{len(rows)} OK.",
          "", "![normGED and depth](normged_depth.png)"]
    (out / "report.md").write_text("\n".join(L) + "\n")
    return L[:-2]


def make_figure(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    x = np.arange(len(rows))
    names = [f"{r['circuit']}" for r in rows]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(max(12, 0.62 * len(rows) + 3), 8.6), dpi=150,
                                 sharex=True)
    fig.patch.set_facecolor("#fcfcfb")
    for ax in (a1, a2):
        ax.set_facecolor("#fcfcfb")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color="#e8e7e2", linewidth=0.6); ax.set_axisbelow(True)
    ys = [r["normGED"] for r in rows]
    bars = a1.bar(x, ys, 0.62, color="#eb6834", edgecolor="#fcfcfb")
    for b, y in zip(bars, ys):
        a1.text(b.get_x() + b.get_width() / 2, y, f"{y:.2f}", ha="center", va="bottom",
                fontsize=6.5, color="#52514e")
    a1.set_ylim(0, 1.1)
    a1.set_title("Q-MaskCut normGED (Eq. 14)", fontsize=10.5, loc="left")
    series = [("original", "depth_orig", "#2a78d6"),
              ("Q-MaskCut, uncut", "depth_qmc_uncut", "#c3c2b7"),
              ("Q-MaskCut, cut (largest fragment)", "depth_qmc_cut", "#eb6834")]
    w = 0.8 / len(series)
    for j, (lab, key, col) in enumerate(series):
        a2.bar(x - 0.4 + w * (j + 0.5), [r[key] for r in rows], w * 0.92, color=col, label=lab,
               edgecolor="#fcfcfb", linewidth=0.8)
    a2.set_yscale("log")
    a2.set_title(f"executed depth on {cfg.BACKEND} (log scale)", fontsize=10.5, loc="left")
    a2.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper left")
    a2.set_xticks(x, names, rotation=40, ha="right", fontsize=8)
    if any(r["n"] > 27 for r in rows) and any(r["n"] <= 27 for r in rows):
        split = next(i for i, r in enumerate(rows) if r["n"] > 27) - 0.5
        for ax in (a1, a2):
            ax.axvline(split, color="#c3c2b7", linewidth=1, linestyle="--")
        a1.text(split + 0.15, 1.04, f"34-40 qubits: fragments <= {MAX_FRAGMENT_WIDTH} wires, "
                f"41-140: <= {WIDE_FRAGMENT_WIDTH}",
                fontsize=7.5, color="#52514e")
    fig.tight_layout()
    fig.savefig(out / "normged_depth.png", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    cfg.update(**SETTINGS)
    main()
