#!/usr/bin/env python3
"""Tier 2: structure and cost only (no simulation), circuits from 3 to 140 qubits, compiled for FakeMarrakesh.

Per circuit and method (original, Q-MaskCut, inverse / composite / delayed gates, ECQCO where an output
exists): normGED and GED_2Q vs the compiled original, compiled depth, gates, 2-qubit gates, estimated SWAPs,
qubits, run time. Q-MaskCut: key (best-TVD key only up to 14 qubits, else random), 2 decoys + 2 padlocks,
H on decoys, CutQC cut (2 balanced fragments up to 40 wires, fragments of at most 27 wires above),
RX pair at the cut, compiled level 3 best of 8, time-aware twirl + no-pattern DD; the fragment that runs
longest is reported, plus the whole protected circuit (for normGED, as in the paper).
Coupling map: good qubits (readout <= 5%, sx <= 1%, 2q <= 5%); circuits wider than the good region (94 on
FakeMarrakesh) use every working qubit, marked in `coupling`.

    .venv/bin/python tier2_structure.py --run runs/tier1_tier2
"""
from __future__ import annotations
import argparse, json, logging, math, os, random, sys, time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("QISKIT_PARALLEL", "FALSE")
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]
T2 = ["bv_n3", "bv_n4", "bell_n4", "qpe_n9", "adder_n10", "seca_n11", "multiply_n13", "bv_n14", "qf21_n15", "qec9xz_n17",
      "bv_n19", "cat_state_n22", "ghz_state_n23", "knn_n25", "ising_n26", "wstate_n27", "adder_n28", "bv_n30", "knn_n31",
      "ising_n34", "cat_n35", "ghz_n40", "ising_n42", "dj_n50", "adder_n64", "cat_n65", "ising_n66", "bv_n70", "dj_n70",
      "ghz_n78", "adder_n118", "ghz_n127", "cat_n130", "bv_n140"]
BACKEND = "structure_FakeMarrakesh"
A = None


def _init(a):
    global A
    A = a


def one(name):
    import numpy as np
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import HGate, RXGate
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
    import new_workflow_test as W
    from qmaskcut.masking import protect_general
    from qmaskcut.cutting import cutqc as C
    from qmaskcut.utils.circuits import load, two_q_only, unmeasured
    from qmaskcut.metrics.normged import deterministic_normged as ng
    from qmaskcut.baselines.obfusqate import OBFUSQATE
    from qmaskcut import config as cfg
    from qmaskcut_eval import circuit_path, good_map, ecqco_circuit
    from qexp_store import RunStore

    W.BEST_OF = A.best_of
    want = lambda m: A.methods is None or m in A.methods                       # noqa: E731
    st = RunStore(A.run); t0 = time.time(); rows = []
    try:
        base = load(name, str(circuit_path(name))); n = base.num_qubits
        info_row = {"circuit": name, "backend": BACKEND, "n": n, "gates": base.size(),
                    "two_q": sum(1 for i in base.data if i.operation.num_qubits == 2)}
        target = FakeMarrakesh().target
        gm = good_map(target); region = len({q for e in gm.get_edges() for q in e})
        wide = n + A.decoys > region
        cmap = W.working_map(target) if wide else gm
        info_row["coupling"] = "all working qubits" if wide else f"good qubits ({region})"
        dur = lambda t: round(t.estimate_duration(target, unit="u"), 3)                       # noqa: E731
        n2 = lambda g: sum(1 for i in g.data if i.operation.num_qubits == 2)                  # noqa: E731
        basis = [W.two_q(target), "rz", "sx", "x", "id"]

        def swaps(logical, g):
            free = transpile(logical, basis_gates=basis, optimization_level=3, seed_transpiler=W.SEED)
            return max(0, round((n2(g) - n2(free)) / 3))

        def g2(a, b):
            ta, tb = two_q_only(a), two_q_only(b)
            return 0.0 if ta.size() == 0 and tb.size() == 0 else ng(ta, tb)

        def compile_(qc):
            m = qc.copy(); m.measure_all(); return W.best(m, target, cmap)

        co = compile_(base); cog = W.gates_only(co)
        if not st.has_circuit(BACKEND, name, "original", "compiled"):
            st.save_circuit(BACKEND, name, "original", "compiled", co)
        rows.append(dict(info_row, method="original", in_subset=want("original"), normged=0.0, ged2q=0.0, depth=cog.depth(), gates_compiled=cog.size(),
                         two_q_compiled=n2(cog), swaps=swaps(base, cog), qubits=n, time_us=dur(co), status="ok"))

        schemes = []
        for tech, make in OBFUSQATE.items():
            if not want(tech):
                continue
            if tech in A.skip:
                rows.append(dict(info_row, method=tech, status="not computed: circuit too large (normGED did not finish in 10 h)"))
                continue
            random.seed(cfg.SEED)
            try:
                schemes.append((tech, unmeasured(make(base.copy()))))
            except Exception as e:                                       # noqa: BLE001
                rows.append(dict(info_row, method=tech, status=f"FAIL {type(e).__name__}: {str(e)[:100]}"))
        if want("ECQCO"):
            ed = ROOT / "docs/ecqco_outputs" / name
            e = QuantumCircuit.from_qasm_file(str(ed)) if ed.exists() else (None if n > 30 else ecqco_circuit(name, base))
            if e is not None:
                schemes.append(("ECQCO", unmeasured(e)))
            else:
                rows.append(dict(info_row, method="ECQCO", status="N/A (no ECQCO output for this circuit)"))
        for tech, obf in schemes:
            try:
                cs = compile_(obf); g = W.gates_only(cs)
                rows.append(dict(info_row, method=tech, normged=ng(cog, g), ged2q=g2(cog, g), depth=g.depth(),
                                 gates_compiled=g.size(), two_q_compiled=n2(g), swaps=swaps(obf, g), qubits=obf.num_qubits,
                                 time_us=dur(cs), status="ok"))
            except Exception as ex:                                      # noqa: BLE001
                rows.append(dict(info_row, method=tech, status=f"FAIL {type(ex).__name__}: {str(ex)[:100]}"))

        # ---- Q-MaskCut
        if not want("Q-MaskCut"):
            raise StopIteration
        prot, tags, info = protect_general(base, n_decoy=A.decoys, n_padlock=A.padlocks, reps=2, seed=A.seed, n_twirl="random")
        lab = QuantumCircuit(*prot.qregs)
        for ins, tg in zip(prot.data, tags):
            if tg != "dd_pad":
                lab.append(ins.operation, ins.qubits)
        for d in info["decoy"]:
            lab.append(HGate(), [d])
        N = lab.num_qubits

        def mask(t, s):
            return W.mask_fill(W.twirl_smart(t, target, s), target, s, max_pairs=A.dd_pairs)

        whole = mask(compile_(lab), A.seed); wg = W.gates_only(whole)
        q = dict(info_row, method="Q-MaskCut", key_mode="random", normged=ng(cog, wg), ged2q=g2(cog, wg),
                 whole_depth=wg.depth(), qubits=N)
        pieces = C.independent_pieces(lab)            # wires that never interact are separate circuits already
        main = pieces[0]; piece = C.piece_circuit(lab, main) if len(pieces) > 1 else lab
        P = piece.num_qubits
        if P <= 40:
            try:
                cq, _ = C.find_balanced_cut(piece, 2, max_cuts=2, allow_extra_cut=True)
            except Exception:                                            # noqa: BLE001
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
        others = [C.piece_circuit(lab, p) for p in pieces[1:]]          # each runs as its own small job
        jobs = subs + others
        frs = [mask(compile_(s), A.seed + k) for k, s in enumerate(jobs)]
        for k, f in enumerate(frs[:len(subs)]):
            if not st.has_circuit(BACKEND, name, "Q-MaskCut", f"fragment_{k}_compiled"):
                st.save_circuit(BACKEND, name, "Q-MaskCut", f"fragment_{k}_compiled", f)
        big = max(range(len(frs)), key=lambda k: frs[k].estimate_duration(target, unit="u")); bg = W.gates_only(frs[big])
        q.update(cuts=cq.cut_solution["num_cuts"], widths="+".join(map(str, C.widths(cq))) +
                 (f" (+{len(others)} uncut pieces, {sum(o.num_qubits for o in others)} wires)" if others else ""),
                 fragments=len(subs), depth=bg.depth(), gates_compiled=bg.size(), two_q_compiled=n2(bg),
                 swaps=swaps(jobs[big], bg), time_us=dur(frs[big]), max_fragment_width=max(C.widths(cq)))
        q["status"] = "ok"; rows.append(q)
    except StopIteration:
        pass
    except Exception as e:                                               # noqa: BLE001
        rows.append({"circuit": name, "backend": BACKEND, "method": "Q-MaskCut", "status": f"FAIL {type(e).__name__}: {str(e)[:200]}"})
    for r in rows:
        r["seconds"] = round(time.time() - t0, 1)
    if A.methods is not None:                                            # a subset: own file, no original row
        rows = [r for r in rows if r.get("method") != "original"]
    st.save_metrics(BACKEND, name + ("" if A.methods is None else "__" + "-".join(m.replace(" ", "") for m in A.methods)), rows)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True); ap.add_argument("--circuits", nargs="*")
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--best-of", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--decoys", type=int, default=2)
    ap.add_argument("--padlocks", type=int, default=2); ap.add_argument("--dd-pairs", type=int, default=8)
    ap.add_argument("--resume", action="store_true", help="skip circuits that already have tier-2 metrics")
    ap.add_argument("--skip", nargs="*", default=[], help="methods marked 'not computed' instead of run")
    ap.add_argument("--methods", nargs="*", default=None, help="only these methods (saved to <circuit>__<methods>.json)")
    a = ap.parse_args()
    names = a.circuits or T2
    if a.resume:
        names = [c for c in names if not (Path(a.run) / "metrics" / BACKEND / f"{c}.json").exists()]
    log = logging.getLogger("tier2"); log.setLevel(logging.INFO); log.propagate = False
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for h in (logging.FileHandler(Path(a.run) / f"log_tier2_{time.strftime('%Y%m%d_%H%M%S')}.log"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt); log.addHandler(h)
    log.info("tier 2 (structure, FakeMarrakesh): %d circuits", len(names))
    with Pool(a.workers, initializer=_init, initargs=(a,)) as pool:
        for rows in pool.imap_unordered(one, sorted(names, key=lambda c: int(c.rsplit("_n", 1)[1]))):
            for r in rows:
                if r.get("status") == "ok":
                    log.info("%-14s %-16s normGED %.3f GED_2Q %.3f depth %s time %s us%s", r["circuit"], r["method"], r["normged"],
                             r["ged2q"], r["depth"], r["time_us"], f"  widths {r['widths']}" if r["method"] == "Q-MaskCut" else "")
                else:
                    log.info("%-14s %-16s %s", r.get("circuit"), r.get("method"), r.get("status"))
    log.info("done")


if __name__ == "__main__":
    main()
