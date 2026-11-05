#!/usr/bin/env python3
"""Recompilation check: does Q-MaskCut's after-compilation masking survive when the provider
recompiles what it receives? The submitted circuit has its delays removed and is transpiled again by Qiskit at
optimisation level 3 for the same target, as a provider would routinely do.

normGED vs the compiled original, on the whole protected circuit (key, decoys, padlocks, decoy H; the paper's normGED):
  before masking   compiled protected circuit
  after masking    + time-aware Pauli twirl + no-pattern DD (the value in the paper)
  recompiled       the masked circuit, delays removed, transpiled again at level 3
and, as a reference, the compiled original recompiled the same way. On the submitted fragments: gates and normGED
against the unmasked compiled fragment, after masking and after recompiling.

Tier 2 (23 hardware circuits, FakeMarrakesh): same protection, compilation and seeds as tier2_structure.py.
BV-3 (FakeMontrealV2): same inputs and compilation as trackb_compiled_normged.py, uniform random key.

    .venv/bin/python recompile_check.py --run runs/recompile            -> <run>/tier2/<circuit>.json
    .venv/bin/python recompile_check.py --run runs/recompile --bv3      -> <run>/bv3.json
    .venv/bin/python recompile_check.py --run runs/recompile --baselines -> <run>/baselines/<circuit>.json
"""
from __future__ import annotations
import argparse, json, math, os, random, sys, time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("QISKIT_PARALLEL", "FALSE")
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]
HW23 = ["bv_n3", "bv_n4", "bell_n4", "qpe_n9", "adder_n10", "seca_n11", "multiply_n13", "bv_n14", "qf21_n15", "qec9xz_n17",
        "bv_n19", "cat_state_n22", "ghz_state_n23", "knn_n25", "ising_n26", "ising_n34", "ghz_n40", "ising_n42", "dj_n50",
        "cat_n65", "dj_n70", "ghz_n78", "ghz_n127"]
A = None


def _init(a):
    global A
    A = a


def no_delays(c):
    o = c.copy_empty_like()
    for i in c.data:
        if i.operation.name not in ("delay", "barrier"):
            o.append(i.operation, i.qubits, i.clbits)
    return o


def labelled(base, seed):
    """Protected circuit before cutting: key, decoys, padlocks, decoy H (as in tier2_structure.py)."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import HGate
    from qmaskcut.masking import protect_general
    prot, tags, info = protect_general(base, n_decoy=2, n_padlock=2, reps=2, seed=seed, n_twirl="random")
    lab = QuantumCircuit(*prot.qregs)
    for ins, tg in zip(prot.data, tags):
        if tg != "dd_pad":
            lab.append(ins.operation, ins.qubits)
    for d in info["decoy"]:
        lab.append(HGate(), [d])
    return lab, info


def tier2(name):
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import RXGate
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
    import new_workflow_test as W
    from qmaskcut.cutting import cutqc as C
    from qmaskcut.utils.circuits import load
    from qmaskcut.metrics.normged import deterministic_normged as ng
    from qmaskcut_eval import circuit_path, good_map

    W.BEST_OF = 8
    t0 = time.time()
    base = load(name, str(circuit_path(name))); n = base.num_qubits
    target = FakeMarrakesh().target
    gm = good_map(target); region = len({q for e in gm.get_edges() for q in e})
    cmap = W.working_map(target) if n + 2 > region else gm
    g = W.gates_only
    compile_ = lambda qc: W.best(qc.measure_all(inplace=False), target, cmap)                        # noqa: E731
    mask = lambda t, s: W.mask_fill(W.twirl_smart(t, target, s), target, s, max_pairs=8)             # noqa: E731
    recompile = lambda c: transpile(no_delays(c), target=target, optimization_level=3, seed_transpiler=W.SEED)  # noqa: E731
    recompile_fixed = lambda c: transpile(no_delays(c), target=target, optimization_level=3, seed_transpiler=W.SEED,  # noqa: E731
                                          initial_layout=list(range(c.num_qubits)))

    co = compile_(base); cog = g(co)
    lab, _ = labelled(base, A.seed)
    unm = compile_(lab); msk = mask(unm, A.seed); rec = recompile(msk)
    row = {"circuit": name, "n": n, "backend": "FakeMarrakesh",
           "normged_before_masking": ng(cog, g(unm)), "normged_after_masking": ng(cog, g(msk)),
           "normged_recompiled": ng(cog, g(rec)), "normged_original_recompiled": ng(cog, g(recompile(co))),
           "normged_recompiled_fixed_layout": ng(cog, g(recompile_fixed(msk))),
           "normged_original_recompiled_fixed_layout": ng(cog, g(recompile_fixed(co))),
           "gates_before_masking": g(unm).size(), "gates_after_masking": g(msk).size(), "gates_recompiled": g(rec).size(),
           "depth_before_masking": g(unm).depth(), "depth_after_masking": g(msk).depth(), "depth_recompiled": g(rec).depth()}

    # submitted fragments (same cut and rotation pairs as tier2_structure.py)
    pieces = C.independent_pieces(lab)
    piece = C.piece_circuit(lab, pieces[0]) if len(pieces) > 1 else lab
    P = piece.num_qubits
    if P <= 40:
        try:
            cq, _ = C.find_balanced_cut(piece, 2, max_cuts=2, allow_extra_cut=True)
        except Exception:                                                # noqa: BLE001
            cq, _ = C.cut_to_width(piece, math.ceil((P + 2) / 2) + 1)
    else:
        cq, _ = C.cut_to_width(piece, 27)
    r = random.Random(1000 + A.seed); subs = [s.copy() for s in cq.cut_solution["subcircuits"]]
    pre = {i: [] for i in range(len(subs))}
    for w, path in C.cpm_index(cq).items():
        for (su, qu), (sd, qd) in zip(path, path[1:]):
            t = r.uniform(0, 2 * math.pi); subs[su].append(RXGate(t), [qu]); pre[sd].append((qd, -t))
    for s, items in pre.items():
        if items:
            h = QuantumCircuit(*subs[s].qregs)
            for qd, t in items:
                h.append(RXGate(t), [qd])
            subs[s] = h.compose(subs[s])
    frags = []
    for k, s in enumerate(subs):
        fu = compile_(s); fm = mask(fu, A.seed + k); fr = recompile(fm)
        frags.append({"fragment": k, "width": s.num_qubits, "gates_before_masking": g(fu).size(),
                      "gates_after_masking": g(fm).size(), "gates_recompiled": g(fr).size(),
                      "normged_masked_vs_unmasked": ng(g(fu), g(fm)), "normged_recompiled_vs_unmasked": ng(g(fu), g(fr))})
    row["fragments"] = frags
    row["seconds"] = round(time.time() - t0, 1)
    out = Path(A.run) / "tier2" / f"{name}.json"; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, indent=1))
    return row


def baselines(name):
    """The other methods, built and compiled as in tier2_structure.py, then recompiled the same way as Q-MaskCut."""
    from qiskit import QuantumCircuit, transpile
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
    import new_workflow_test as W
    from qmaskcut.utils.circuits import load, unmeasured
    from qmaskcut.metrics.normged import deterministic_normged as ng
    from qmaskcut.baselines.obfusqate import OBFUSQATE
    from qmaskcut import config as cfg
    from qmaskcut_eval import circuit_path, good_map, ecqco_circuit

    W.BEST_OF = 8
    t0 = time.time()
    base = load(name, str(circuit_path(name))); n = base.num_qubits
    target = FakeMarrakesh().target
    gm = good_map(target); region = len({q for e in gm.get_edges() for q in e})
    cmap = W.working_map(target) if n + 2 > region else gm
    g = W.gates_only
    compile_ = lambda qc: W.best(qc.measure_all(inplace=False), target, cmap)                        # noqa: E731
    recompile = lambda c: transpile(no_delays(c), target=target, optimization_level=3, seed_transpiler=W.SEED)  # noqa: E731
    recompile_fixed = lambda c: transpile(no_delays(c), target=target, optimization_level=3, seed_transpiler=W.SEED,  # noqa: E731
                                          initial_layout=list(range(c.num_qubits)))
    co = compile_(base); cog = g(co)
    schemes = [("original", base)]
    for tech, make in OBFUSQATE.items():
        random.seed(cfg.SEED)
        schemes.append((tech, unmeasured(make(base.copy()))))
    ed = ROOT / "docs/ecqco_outputs" / name
    e = QuantumCircuit.from_qasm_file(str(ed)) if ed.exists() else (None if n > 30 else ecqco_circuit(name, base))
    if e is not None:
        schemes.append(("ECQCO", unmeasured(e)))
    rows = []
    for tech, obf in schemes:
        cs = co if tech == "original" else compile_(obf)
        rows.append({"circuit": name, "n": n, "method": tech, "normged_compiled": ng(cog, g(cs)),
                     "normged_recompiled": ng(cog, g(recompile(cs))),
                     "normged_recompiled_fixed_layout": ng(cog, g(recompile_fixed(cs))),
                     "gates_compiled": g(cs).size(), "gates_recompiled_fixed_layout": g(recompile_fixed(cs)).size()})
    out = Path(A.run) / "baselines" / f"{name}.json"; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"circuit": name, "rows": rows, "seconds": round(time.time() - t0, 1)}, indent=1))
    return rows


def bv3(label):
    """One BV-3 input, as trackb_compiled_normged.py, but with a uniform random key (first draw of the seed)."""
    import numpy as np
    from qiskit import QuantumCircuit, transpile
    from qiskit_ibm_runtime.fake_provider import FakeMontrealV2
    import new_workflow_test as W
    from qmaskcut.utils.circuits import load
    from qmaskcut.metrics.normged import deterministic_normged as ng
    from qmaskcut_eval import circuit_path, good_map
    target = FakeMontrealV2().target; cmap = good_map(target)
    tp = lambda qc: transpile(qc, basis_gates=["cx", "rz", "sx", "x", "id"], coupling_map=cmap, optimization_level=3,  # noqa: E731
                              seed_transpiler=42)
    recompile = lambda c: transpile(no_delays(c), target=target, optimization_level=3, seed_transpiler=42)  # noqa: E731
    recompile_fixed = lambda c: transpile(no_delays(c), target=target, optimization_level=3, seed_transpiler=42,  # noqa: E731
                                          initial_layout=list(range(c.num_qubits)))
    base0 = load("bv_n3", str(circuit_path("bv_n3"))); n = base0.num_qubits
    it, k = int(label[2:4]), int(label[7:9])
    rng = np.random.default_rng(1000 * it + k); qc = QuantumCircuit(n)
    for q in range(n):
        qc.u(*rng.uniform(0, 2 * np.pi, 3), q)
    qc.compose(base0, inplace=True)
    g = W.gates_only
    co = tp(qc.measure_all(inplace=False)); cog = g(co)
    sd = 100 * it                                    # protection seed = iteration, as trackb_bv3.py
    lab, info = labelled(qc, sd)
    unm = tp(lab.measure_all(inplace=False))
    msk = W.mask_fill(W.twirl_smart(unm, target, sd), target, sd, max_pairs=8)
    rec = recompile(msk)
    return {"label": label, "key": info["flip"], "normged_before_masking": ng(cog, g(unm)),
            "normged_after_masking": ng(cog, g(msk)), "normged_recompiled": ng(cog, g(rec)),
            "normged_original_recompiled": ng(cog, g(recompile(co))),
            "normged_recompiled_fixed_layout": ng(cog, g(recompile_fixed(msk))),
            "normged_original_recompiled_fixed_layout": ng(cog, g(recompile_fixed(co))),
            "gates_before_masking": g(unm).size(), "gates_after_masking": g(msk).size(), "gates_recompiled": g(rec).size()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True); ap.add_argument("--circuits", nargs="*")
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bv3", action="store_true")
    ap.add_argument("--baselines", action="store_true", help="recompile the other methods too -> <run>/baselines/<circuit>.json")
    a = ap.parse_args(); Path(a.run).mkdir(parents=True, exist_ok=True)
    if a.baselines:
        names = a.circuits or HW23
        with Pool(a.workers, initializer=_init, initargs=(a,)) as p:
            for rows in p.imap_unordered(baselines, sorted(names, key=lambda c: int(c.rsplit("_n", 1)[1]))):
                for r in rows:
                    print(f"{r['circuit']:14s} {r['method']:16s} compiled {r['normged_compiled']:.3f}  recompiled "
                          f"{r['normged_recompiled']:.3f} / fixed {r['normged_recompiled_fixed_layout']:.3f}", flush=True)
        return
    if a.bv3:
        labels = [f"it{i:02d}_in{j:02d}" for i in range(20) for j in range(10)]
        with Pool(a.workers, initializer=_init, initargs=(a,)) as p:
            res = p.map(bv3, labels)
        Path(a.run, "bv3.json").write_text(json.dumps(res, indent=0))
        import statistics as S
        for c in ("normged_before_masking", "normged_after_masking", "normged_recompiled", "normged_original_recompiled",
                  "normged_recompiled_fixed_layout", "normged_original_recompiled_fixed_layout"):
            v = [r[c] for r in res]
            print(f"bv3 {c:28s} {S.mean(v):.3f} ± {S.stdev(v):.3f}", flush=True)
        return
    names = a.circuits or HW23
    with Pool(a.workers, initializer=_init, initargs=(a,)) as p:
        for row in p.imap_unordered(tier2, sorted(names, key=lambda c: int(c.rsplit("_n", 1)[1]))):
            print(f"{row['circuit']:14s} before {row['normged_before_masking']:.3f}  masked {row['normged_after_masking']:.3f}  "
                  f"recompiled {row['normged_recompiled']:.3f} / fixed {row['normged_recompiled_fixed_layout']:.3f}  (original recompiled {row['normged_original_recompiled']:.3f})  "
                  f"gates {row['gates_before_masking']}/{row['gates_after_masking']}/{row['gates_recompiled']}  {row['seconds']} s",
                  flush=True)


if __name__ == "__main__":
    main()
