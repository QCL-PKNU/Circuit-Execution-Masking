#!/usr/bin/env python3
"""Prepare the hardware campaign WITHOUT submitting anything (no QPU time is used).

For every circuit: the same protection and the same CutQC cut as main.py (MAX_CUTS = 2, no
extra cut), the uncut baseline plus every fragment instance, compiled for the real device.
Everything is saved so that hw_submit.py later sends exactly these circuits:

  results/hw3_prepared/<backend>/<name>.pkl   circuits, partition, key, labels (to submit
                                              and to reconstruct with CutQC afterwards)
  results/hw3_prepared/<backend>/plan.csv     one row per circuit: sizes, depth, QPU estimate
  results/hw3_prepared/<backend>/plan.md      the same as a table, plus the quota left

QPU estimate: fitted on the hw2 campaign (ibm_marrakesh, 2048 shots), 16-20 s for 19-25
circuits and 56 s for 92: seconds ~= 4 + circuits x shots x (2.7e-4 + depth x 6e-8), the
second term for long circuits (about 60 ns per layer on Heron).

    .venv/bin/python scripts/hw_prepare.py                     # ibm_marrakesh, 4096 shots
    .venv/bin/python scripts/hw_prepare.py --backend ibm_fez --shots 2048
"""
from __future__ import annotations
import argparse, csv, pickle, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qiskit import transpile
from qiskit_ibm_runtime import QiskitRuntimeService
from qmaskcut import config as cfg
from qmaskcut.utils.circuits import load
from qmaskcut.masking.protect import protect_general
from qmaskcut.cutting import cutqc as C
from qmaskcut.metrics.ideal import ideal

# (name, QASMBench path or None for BV 1...1, fragments; None = cut to 20-wire fragments as
# in the 4-140 scaling study). Groups: Table II; FakeMontreal could not finish; the rest of
# the 25-circuit screen (4+ qubits, non-uniform output); timed out in simulation; 34-140.
CASES = [
    ("bv_n4",         None,                                       3),
    ("qpe_n9",        "small/qpe_n9/qpe_n9.qasm",                 2),
    ("adder_n10",     "small/adder_n10/adder_n10.qasm",           2),
    ("seca_n11",      "medium/seca_n11/seca_n11.qasm",            2),
    ("multiply_n13",  "medium/multiply_n13/multiply_n13.qasm",    2),
    ("bv_n14",        "medium/bv_n14/bv_n14.qasm",                2),
    ("qf21_n15",      "medium/qf21_n15/qf21_n15.qasm",            2),
    ("qec9xz_n17",    "medium/qec9xz_n17/qec9xz_n17.qasm",        2),
    ("bigadder_n18",  "medium/bigadder_n18/bigadder_n18.qasm",    2),
    ("bv_n19",        "medium/bv_n19/bv_n19.qasm",                3),
    ("cat_state_n22", "medium/cat_state_n22/cat_state_n22.qasm",  3),
    ("ghz_state_n23", "medium/ghz_state_n23/ghz_state_n23.qasm",  3),
    ("qram_n20",      "medium/qram_n20/qram_n20.qasm",            2),
    ("wstate_n27",    "medium/wstate_n27/wstate_n27.qasm",        2),
    ("bell_n4",       "small/bell_n4/bell_n4.qasm",               2),
    ("cat_state_n4",  "small/cat_state_n4/cat_state_n4.qasm",     2),
    ("error_correctiond3_n5", "small/error_correctiond3_n5/error_correctiond3_n5.qasm", 2),
    ("qaoa_n6",       "small/qaoa_n6/qaoa_n6.qasm",               2),
    ("hhl_n7",        "small/hhl_n7/hhl_n7.qasm",                 2),
    ("sat_n7",        "small/sat_n7/sat_n7.qasm",                 2),
    ("dnn_n8",        "small/dnn_n8/dnn_n8.qasm",                 2),
    ("ising_n10",     "small/ising_n10/ising_n10.qasm",           2),
    ("sat_n11",       "medium/sat_n11/sat_n11.qasm",              2),
    ("multiplier_n15", "medium/multiplier_n15/multiplier_n15.qasm", 2),
    ("dnn_n16",       "medium/dnn_n16/dnn_n16.qasm",              2),
    ("square_root_n18", "medium/square_root_n18/square_root_n18.qasm", 2),
    ("cat_n35",       "large/cat_n35/cat_n35.qasm",               None),
    ("ghz_n40",       "large/ghz_n40/ghz_n40.qasm",               None),
    ("cat_n65",       "large/cat_n65/cat_n65.qasm",               None),
    ("bv_ones_n70",   None,                                       None),
    ("ghz_n78",       "large/ghz_n78/ghz_n78.qasm",               None),
    ("ghz_n127",      "large/ghz_n127/ghz_n127.qasm",             None),
    ("cat_n130",      "large/cat_n130/cat_n130.qasm",             None),
    ("bv_ones_n140",  None,                                       None),
]
# Not on the list, and why: qft_n4 / qft_n18 (uniform output: no fidelity to measure);
# knn_n25, swap_test_n25 (one measured qubit); ising_n26, ising_n34 (no usable reference);
# vqe_n24, bwt_n21, factor247_n15 (millions of gates); grover_n2, toffoli_n3, fredkin_n3,
# wstate_n3 (under 4 qubits); QASMBench bv_n70 / bv_n140 (random secret: the circuit falls
# into independent pieces; the scaling study uses BV with secret 1...1, as here); inverseqft_n4, shor_n5, cc_n12 (classically controlled gates);
# simon_n6, bb84_n8, lpn_n5 (already independent pieces); vqe_uccsd_n8, hhl_n10, hhl_n14
# (QASM does not parse).
MAX_WIDTH = 20                     # fragment cap for the 34-140 circuits, as main-benchmark.py
MAX_CUTS, DECOYS, PADLOCKS, SEED, OPT = 2, 2, 2, 42, 1      # as main.py


def reference(name, base):
    """Ideal output, little-endian keys. Clifford circuits (GHZ, cat, BV at any width) by
    stabilizer simulation, so a QASMBench BV file's own random secret is used, not 1...1."""
    from qiskit.quantum_info import StabilizerState
    try:
        p = StabilizerState(base).probabilities_dict()
        return {k[::-1]: float(v) for k, v in p.items()}, "stabilizer"
    except Exception:              # noqa: BLE001 - not Clifford
        pass
    return ideal(name, base)


def qpu_estimate(depths: list[int], shots: int) -> float:
    return 4 + sum(shots * (2.7e-4 + d * 6e-8) for d in depths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ibm_marrakesh")
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--best-of", type=int, default=1, help="noise-aware placement: best of N compiles")
    ap.add_argument("--opt", type=int, default=OPT, help="transpiler optimization level")
    ap.add_argument("--max-width", type=int, default=MAX_WIDTH,
                    help="fragment cap for the wide circuits (fragments = None in CASES)")
    a = ap.parse_args()
    out = ROOT / "results" / "hw3_prepared" / a.backend
    out.mkdir(parents=True, exist_ok=True)

    svc = QiskitRuntimeService(name="pknu")              # read-only: backend target + quota
    backend = svc.backend(a.backend)
    usage = svc.usage()
    left = usage["usage_limit_seconds"] - usage["usage_consumed_seconds"]
    print(f"backend {a.backend} ({backend.num_qubits} qubits), {a.shots} shots, "
          f"QPU left this period: {left} s (of {usage['usage_limit_seconds']} s)", flush=True)

    rows = []
    for name, rel, pieces in CASES:
        if a.only and name not in a.only:
            continue
        t = time.time()
        try:
            row = prepare(name, rel, pieces, backend, a.shots, out, a.best_of, a.opt, a.max_width)
        except Exception as e:          # noqa: BLE001 - one circuit must not stop the plan
            print(f"{name:14s} SKIPPED: {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        rows.append(row)
        print(f"{name:14s} n={row['n']:3d} pieces {row['pieces']:10s} K={row['K']} "
              f"{row['circuits_total']:3d} circuits | depth base {row['depth_baseline']:5d}, "
              f"fragment max {row['depth_fragment_max']:5d} | ~{row['qpu_est_s']:5.1f} s QPU "
              f"| {time.time() - t:.0f}s", flush=True)
    write_plan(rows, out, a, left, usage)


def log_esp(tc, target) -> float:
    """log of the estimated success probability: sum of log(1 - error) over every gate and
    measurement, from the live calibration (higher is better)."""
    import math
    s = 0.0
    for ins in tc.data:
        op = ins.operation.name
        if op in ("barrier", "delay") or op not in target.operation_names:
            continue
        qs = tuple(tc.find_bit(q).index for q in ins.qubits)
        prop = target[op].get(qs)
        if prop is not None and prop.error is not None:
            s += math.log(max(1.0 - prop.error, 1e-9))
    return s


def best_compile(circ, backend, best_of, opt, layout=None):
    """Noise-aware placement: compile best_of times (different seeds), keep the highest ESP.
    With `layout` given, the physical qubits are fixed and only routing varies."""
    best = None
    for k in range(best_of):
        tc = transpile(circ, backend, optimization_level=opt, seed_transpiler=SEED + k,
                       initial_layout=layout)
        sc = log_esp(tc, backend.target)
        if best is None or sc > best[1]:
            best = (tc, sc)
    return best


def prepare(name, rel, pieces, backend, shots, out, best_of=1, opt=OPT, max_width=MAX_WIDTH):
    """Protect, cut, compile; save everything hw_submit.py needs. Returns the plan row."""
    base = load(name, rel)
    n = base.num_qubits
    ref, how = reference(name, base)
    prot, tags, info = protect_general(base, n_decoy=DECOYS, n_padlock=PADLOCKS,
                                       reps=2, seed=SEED)
    if pieces is None:
        cq, _ = C.cut_to_width(prot, max_width)
    else:
        cq, _ = C.find_balanced_cut(prot, pieces, max_cuts=MAX_CUTS, allow_extra_cut=False)
    labels, frags = C.instance_circuits(cq)
    bq = base.copy(); bq.measure_all()
    fr = []
    for f in frags:
        f = f.copy()
        if f.num_clbits == 0:
            f.measure_all()
        fr.append(f)
    if best_of <= 1:
        tb = transpile(bq, backend, optimization_level=opt, seed_transpiler=SEED)
        tf = transpile(fr, backend, optimization_level=opt, seed_transpiler=SEED)
    else:
        tb, _ = best_compile(bq, backend, best_of, opt)
        tf = [None] * len(fr)
        for s in sorted({lab[0] for lab in labels}):        # one layout per fragment
            idx = [j for j, lab in enumerate(labels) if lab[0] == s]
            rep_tc, _ = best_compile(fr[idx[0]], backend, best_of, opt)
            lay = rep_tc.layout.initial_index_layout(filter_ancillas=True)[:fr[idx[0]].num_qubits]
            for j in idx:
                tf[j] = transpile(fr[j], backend, optimization_level=opt, seed_transpiler=SEED,
                                  initial_layout=lay)
    n_circ = 1 + len(tf)
    row = {"circuit": name, "n": n, "N": prot.num_qubits,
           "pieces": "+".join(map(str, C.widths(cq))), "K": cq.cut_solution["num_cuts"],
           "fragment_instances": len(tf), "circuits_total": n_circ, "shots": shots,
           "depth_baseline": tb.depth(), "twoq_baseline": tb.num_nonlocal_gates(),
           "depth_fragment_max": max(c.depth() for c in tf),
           "twoq_fragment_max": max(c.num_nonlocal_gates() for c in tf),
           "ideal": how if ref is not None else "none",
           "key": info["flip"],
           "placement": f"best of {best_of} compiles by ESP, opt {opt}" if best_of > 1 else f"opt {opt}",
           "log_esp_baseline": round(log_esp(tb, backend.target), 3),
           "log_esp_fragment_min": round(min(log_esp(c, backend.target) for c in tf), 3),
           "qpu_est_s": round(qpu_estimate([tb.depth()] + [c.depth() for c in tf], shots), 1)}
    pickle.dump({"row": row, "base": base, "prot": prot, "tags": tags, "info": info,
                 "subcircuits": cq.cut_solution["subcircuits"],
                 "complete_path_map": cq.cut_solution["complete_path_map"],
                 "labels": labels, "baseline_tx": tb, "fragments_tx": tf,
                 "backend": backend.name, "shots": shots, "seed": SEED},
                open(out / f"{name}.pkl", "wb"))
    return row


def write_plan(rows, out, a, left, usage):
    if (out / "plan.csv").exists():                 # keep circuits prepared in earlier calls
        new = {r["circuit"] for r in rows}
        old = [r for r in csv.DictReader(open(out / "plan.csv")) if r["circuit"] not in new]
        rows = old + rows
    with open(out / "plan.csv", "w", newline="") as f:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    total = sum(float(r["qpu_est_s"]) for r in rows)
    md = [f"# Hardware plan: {a.backend}, {a.shots} shots (prepared, NOT submitted)", "",
          f"QPU left this period: **{left} s** of {usage['usage_limit_seconds']} s. "
          f"Estimated need: **{total:.0f} s** for {len(rows)} circuits.", "",
          "| circuit | n | pieces | K | circuits | depth uncut | depth largest fragment | QPU est (s) |",
          "|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['circuit']} | {r['n']} | {r['pieces']} | {r['K']} | {r['circuits_total']} | "
           f"{r['depth_baseline']} | {r['depth_fragment_max']} | {r['qpu_est_s']} |" for r in rows]
    (out / "plan.md").write_text("\n".join(md) + "\n")
    print(f"\nestimated QPU {total:.0f} s for {len(rows)} circuits; {left} s left. "
          f"Saved to {out.relative_to(ROOT)}/ (nothing submitted)")


if __name__ == "__main__":
    main()
