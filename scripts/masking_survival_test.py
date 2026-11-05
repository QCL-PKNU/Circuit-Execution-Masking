#!/usr/bin/env python3
"""Can idle-time masking survive optimisation level 3? No QPU is used.

The masking sequences multiply to the identity, so level-3 optimisation merges and deletes
them. This test compiles, for each circuit, (1) the original, (2) Q-MaskCut as built, and
(3) Q-MaskCut with a single-qubit barrier after EVERY masking gate (a barrier around a whole
block would not help: the block would still merge to the identity inside it). All compiled
for FakeMarrakesh, level 3, seed 42. Reported: compiled depth and gate counts (what it costs),
and normGED / GED_2Q of the compiled circuit against the compiled original (what survives).

    .venv/bin/python scripts/masking_survival_test.py   -> results/masking_survival/results.csv
"""
from __future__ import annotations
import csv, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
bench = __import__("main-benchmark")
from qmaskcut import config as cfg
cfg.update(**bench.SETTINGS)
from qiskit import QuantumCircuit
from qmaskcut.device import Device
from qmaskcut.masking import protect_general
from qmaskcut.metrics.normged import deterministic_normged
from qmaskcut.utils.circuits import load, two_q_only, n2q

CIRCS = {"bell_n4": "small/bell_n4/bell_n4.qasm", "qpe_n9": "small/qpe_n9/qpe_n9.qasm",
         "bv_n19": "medium/bv_n19/bv_n19.qasm"}


def with_barriers(prot: QuantumCircuit, tags) -> QuantumCircuit:
    """The protected circuit with a one-qubit barrier after every masking ('dd_pad') gate."""
    out = QuantumCircuit(*prot.qregs, *prot.cregs)
    for ins, tag in zip(prot.data, tags):
        out.append(ins.operation, ins.qubits, ins.clbits)
        if tag == "dd_pad":
            out.barrier(ins.qubits)
    return out


def compiled(dev, qc):
    t = dev.compile(qc).remove_final_measurements(inplace=False)
    plain = QuantumCircuit(*t.qregs)
    for ins in t.data:
        if ins.operation.name != "barrier":
            plain.append(ins.operation, ins.qubits)
    ops = t.count_ops()
    q1 = sum(v for k, v in ops.items() if k in ("rz", "sx", "x"))
    return plain, t.depth(), q1, ops.get("cz", 0)


if __name__ == "__main__":
    dev = Device(cfg.BACKEND)
    rows = []
    for name, path in CIRCS.items():
        base = load(name, path)
        prot, tags, info = protect_general(base, n_decoy=cfg.QMC_DECOYS, n_padlock=cfg.QMC_PADLOCKS,
                                           reps=2, seed=cfg.SEED)
        assert len(tags) == len(prot.data)
        nmask = sum(t == "dd_pad" for t in tags)
        cb, d0, q10, cz0 = compiled(dev, base)
        for variant, qc in (("original", base), ("Q-MaskCut (as built)", prot),
                            ("Q-MaskCut + barriers", with_barriers(prot, tags))):
            t = time.time()
            c, d, q1, cz = compiled(dev, qc)
            ng = 0.0 if variant == "original" else deterministic_normged(cb, c)
            g2 = 0.0 if variant == "original" else deterministic_normged(two_q_only(cb), two_q_only(c))
            rows.append({"circuit": name, "variant": variant, "masking_gates_written": nmask,
                         "compiled_depth": d, "compiled_1q": q1, "compiled_cz": cz,
                         "depth_vs_original": round(d / d0, 2),
                         "extra_1q_vs_original": q1 - q10,
                         "normGED_compiled": round(ng, 4), "GED2Q_compiled": round(g2, 4)})
            print(f"{name:8s} {variant:22s} depth {d:5d} ({d / d0:5.2f}x)  1q {q1:5d} (+{q1 - q10:5d})  "
                  f"cz {cz:4d}  normGED {ng:.3f}  GED2Q {g2:.3f}  {time.time() - t:.0f}s", flush=True)
    out = ROOT / "results/masking_survival"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("wrote", out / "results.csv")
