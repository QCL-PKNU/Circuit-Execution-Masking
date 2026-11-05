#!/usr/bin/env python3
"""Compiled normGED for the BV-3 spec run (the paper's normGED): compiled original vs compiled method circuit, as the
compiler sees it. Q-MaskCut: the whole protected circuit (key, decoys, padlocks, decoy H) compiled at the run's optimization level with the
same seeds as trackb_bv3.py, then time-aware twirl + no-pattern DD (the masking is added after compilation, so the
logical normGED of trackb_bv3.py does not contain it). Baselines: their saved compiled circuits.

    .venv/bin/python trackb_compiled_normged.py --run runs/bv3   -> <run>/normged_compiled.json
"""
import argparse, glob, itertools, json, os, sys
from multiprocessing import Pool
from pathlib import Path
os.environ.setdefault("QISKIT_PARALLEL", "FALSE")
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]
RUN = None
KEY_MODE = "best"


def one(label):
    import numpy as np
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import HGate
    from qiskit.quantum_info import Statevector
    from qiskit_ibm_runtime.fake_provider import FakeMontrealV2
    import new_workflow_test as W
    from qmaskcut.masking import protect_general
    from qmaskcut.utils.circuits import load, two_q_only
    from qmaskcut.metrics.normged import deterministic_normged as ng
    from qmaskcut_eval import circuit_path, good_map
    from qexp_store import RunStore
    st = RunStore(RUN); B = "FakeMontrealV2"
    target = FakeMontrealV2().target; cmap = good_map(target)
    tp = lambda qc: transpile(qc, basis_gates=["cx", "rz", "sx", "x", "id"], coupling_map=cmap, optimization_level=int(json.load(open(f"{RUN}/config.json"))["optimization_level"]),   # noqa: E731
                              seed_transpiler=42)
    base0 = load("bv_n3", str(circuit_path("bv_n3"))); n = base0.num_qubits
    if label == "plain":
        qc = base0.copy(); it = 99
    else:
        it, k = int(label[2:4]), int(label[7:9])
        rng = np.random.default_rng(1000 * it + k); qc = QuantumCircuit(n)
        for q in range(n):
            qc.u(*rng.uniform(0, 2 * np.pi, 3), q)
        qc.compose(base0, inplace=True)
    g = lambda c: W.gates_only(c)                                                            # noqa: E731
    role = lambda m, r: r + "_sched" if st.has_circuit(B, label, m, r + "_sched") else r            # noqa: E731
    co = g(st.load_circuit(B, label, "original", role("original", "run")))

    def g2(a, b):
        ta, tb = two_q_only(a), two_q_only(b)
        return 0.0 if ta.size() == 0 and tb.size() == 0 else ng(ta, tb)
    out = {"label": label}
    for m in ("inverse gates", "composite gates", "delayed gates", "ECQCO"):
        r_ = role(m, "compiler_view")
        if st.has_circuit(B, label, m, r_):
            c = g(st.load_circuit(B, label, m, r_)); out[m] = (ng(co, c), g2(co, c))
    # Q-MaskCut: same key search and seeds as trackb_bv3.py
    ideal = Statevector(qc).probabilities(); ft = {}
    for f in itertools.product([0, 1], repeat=n):
        if any(f):
            kk = np.arange(2 ** n) ^ sum(b << q for q, b in enumerate(f)); ft["".join(map(str, f))] = 0.5 * np.abs(ideal - ideal[kk]).sum()
    best = {x for x, y in ft.items() if y > max(ft.values()) - 1e-9}
    for sd in range(100 * it, 100 * it + (300 if KEY_MODE == "best" else 1)):
        prot, tags, info = protect_general(qc, n_decoy=2, n_padlock=2, reps=2, seed=sd, n_twirl="random")
        if KEY_MODE == "uniform" or info["flip"] in best:
            break
    lab = QuantumCircuit(*prot.qregs)
    for ins, tg in zip(prot.data, tags):
        if tg != "dd_pad":
            lab.append(ins.operation, ins.qubits)
    for d in info["decoy"]:
        lab.append(HGate(), [d])
    whole = W.mask_fill(W.twirl_smart(tp(lab.measure_all(inplace=False)), target, sd), target, sd, max_pairs=8)
    c = g(whole); out["Q-MaskCut"] = (ng(co, c), g2(co, c))
    out["Q-MaskCut_no_masking"] = (lambda c2: (ng(co, c2), g2(co, c2)))(g(tp(lab.measure_all(inplace=False))))
    return out


def main():
    global RUN, KEY_MODE
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--key-mode", choices=["best", "uniform"], default="best", help="as trackb_bv3.py --key-mode")
    a = ap.parse_args(); RUN = a.run; KEY_MODE = a.key_mode
    rows = []
    for f in glob.glob(f"{a.run}/rows_*.json"):
        rows += json.load(open(f))
    labels = sorted({r["label"] for r in rows})
    def init(r, k):
        globals().update(RUN=r, KEY_MODE=k)
    with Pool(a.workers, initializer=init, initargs=(a.run, a.key_mode)) as p:
        res = p.map(one, labels)
    Path(f"{a.run}/normged_compiled.json").write_text(json.dumps(res, indent=0))
    import pandas as pd
    for m in ("Q-MaskCut", "Q-MaskCut_no_masking", "inverse gates", "composite gates", "delayed gates", "ECQCO"):
        v = [r[m][0] for r in res if m in r and r["label"] != "plain"]; v2 = [r[m][1] for r in res if m in r and r["label"] != "plain"]
        print(f"{m:22s} normGED {pd.Series(v).mean():.3f} ± {pd.Series(v).std():.3f}   GED_2Q {pd.Series(v2).mean():.3f}   ({len(v)} runs)")


if __name__ == "__main__":
    main()
