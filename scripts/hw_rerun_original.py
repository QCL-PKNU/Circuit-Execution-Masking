#!/usr/bin/env python3
"""Fair rerun of the UNCUT ORIGINAL on ghz_n127 and cat_n130, plus one readout check.

In the 6 October run, the original's layout for these two circuits used 6 CZ couplers that
ibm_marrakesh's own calibration marked broken (error 1.0) before the job was prepared: the
compiler keeps such couplers in the coupling map, so every one of the 8 candidate layouts
used them. The Q-MaskCut fragments used none. This script compiles the original again on
the chip WITHOUT couplers whose current error is >= 0.5 (best of 8 by estimated success
probability, as hw_prepare.py), and adds a readout check: X on every qubit cat_n130 uses,
then measure, so the all-ones answer can only be lost in readout.

Same settings as hw_submit.py --mitigate: SamplerV2, 2048 shots, dynamical decoupling
(XpXm), gate + measurement twirling. One job.

    .venv/bin/python scripts/hw_rerun_original.py --account pknu3            # compile + check only
    .venv/bin/python scripts/hw_rerun_original.py --account pknu3 --submit   # also run on the QPU
    .venv/bin/python scripts/hw_rerun_original.py --account pknu3 --collect  # fetch + score
"""
from __future__ import annotations
import argparse, json, math, pickle, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from hw_prepare import log_esp

PREP = ROOT / "results/hw3_prepared/ibm_marrakesh"
OUT = ROOT / "results/hw3/ibm_marrakesh_rerun_original"
CIRCS = ["ghz_n127", "cat_n130"]
SHOTS, BEST_OF, SEED, BROKEN = 2048, 8, 42, 0.5
OPTS = {"dynamical_decoupling": {"enable": True, "sequence_type": "XpXm"},
        "twirling": {"enable_gates": True, "enable_measure": True,
                     "num_randomizations": "auto", "shots_per_randomization": "auto"}}


def working_map(backend):
    t = backend.target
    bad = [qs for qs, p in t["cz"].items() if p is not None and p.error is not None and p.error >= BROKEN]
    edges = [qs for qs, p in t["cz"].items() if qs not in bad]
    return CouplingMap(edges), bad


def broken_used(tc, bad):
    bad = {tuple(sorted(e)) for e in bad}
    return sum(1 for i in tc.data if i.operation.name == "cz"
               and tuple(sorted(tc.find_bit(q).index for q in i.qubits)) in bad)


def compile_fair(circ, backend, cmap):
    best = None
    for k in range(BEST_OF):
        tc = transpile(circ, basis_gates=["cz", "rz", "sx", "x", "id"], coupling_map=cmap,
                       optimization_level=3, seed_transpiler=SEED + k)
        sc = log_esp(tc, backend.target)
        if best is None or sc > best[1]:
            best = (tc, sc)
    return best


def readout_check(tc, n):
    """X on every physical qubit tc measures, then measure them in the same clbit order."""
    m = {tc.find_bit(i.clbits[0]).index: tc.find_bit(i.qubits[0]).index
         for i in tc.data if i.operation.name == "measure"}
    qc = QuantumCircuit(tc.num_qubits, n, name="readout_all_ones")
    for c in range(n):
        qc.x(m[c])
    for c in range(n):
        qc.measure(m[c], c)
    return qc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", required=True)
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--collect", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    svc = QiskitRuntimeService(name=a.account)
    backend = svc.backend("ibm_marrakesh")
    if a.collect:
        rec = json.loads((OUT / "job.json").read_text())
        res = svc.job(rec["job"]).result()
        for k, item in enumerate(rec["items"]):
            counts = res[k].data.c.get_counts() if hasattr(res[k].data, "c") else \
                next(iter(res[k].data.values())).get_counts()
            n = item["n"]
            good = {"0" * n, "1" * n} if item["kind"] == "original" else {"1" * n}
            hits = sum(v for b, v in counts.items() if b.replace(" ", "") in good)
            item.update(correct_shots=hits, shots=sum(counts.values()),
                        p_correct=hits / sum(counts.values()))
            print(f"{item['name']:22s} correct shots {hits} of {sum(counts.values())}")
        rec["collected"] = time.strftime("%Y-%m-%d %H:%M:%S")
        (OUT / "job.json").write_text(json.dumps(rec, indent=1))
        sys.exit()
    cmap, bad = working_map(backend)
    print(f"ibm_marrakesh now: {len(bad)} CZ couplers with error >= {BROKEN}, left out")
    circuits, items = [], []
    for c in CIRCS:
        P = pickle.load(open(PREP / f"{c}.pkl", "rb"))
        base = P["base"].copy(); base.measure_all()
        tc, sc = compile_fair(base, backend, cmap)
        old = P["baseline_tx"]
        print(f"{c:9s} new layout: depth {tc.depth():4d}, {tc.count_ops().get('cz', 0):3d} CZ, broken used "
              f"{broken_used(tc, bad)}, log ESP {sc:.2f} | 6 Oct layout: depth {old.depth()}, broken used "
              f"{broken_used(old, bad)}, log ESP {log_esp(old, backend.target):.2f}")
        circuits.append(tc)
        items.append({"name": f"{c}_original", "kind": "original", "circuit": c,
                      "n": P["base"].num_qubits, "depth": tc.depth(), "log_esp": round(sc, 3),
                      "broken_couplers_used": broken_used(tc, bad)})
    n130 = items[-1]["n"]                       # cat_n130, the last circuit compiled
    rc = readout_check(circuits[-1], n130)
    circuits.append(rc)
    items.append({"name": "cat_n130_readout_all_ones", "kind": "readout", "circuit": "cat_n130",
                  "n": n130, "depth": rc.depth()})
    print(f"readout check: X then measure on the {items[-1]['n']} qubits of cat_n130's new layout")
    with open(OUT / "circuits.pkl", "wb") as f:
        pickle.dump({"circuits": circuits, "items": items, "broken": bad}, f)
    if not a.submit:
        print("compile only; add --submit to run")
        sys.exit()
    job = SamplerV2(mode=backend, options=OPTS).run(circuits, shots=SHOTS)
    rec = {"job": job.job_id(), "account": a.account, "backend": "ibm_marrakesh",
           "submitted": time.strftime("%Y-%m-%d %H:%M:%S"), "shots": SHOTS,
           "mitigation": OPTS, "items": items, "broken_couplers_left_out": len(bad)}
    (OUT / "job.json").write_text(json.dumps(rec, indent=1))
    print("submitted job", job.job_id())
