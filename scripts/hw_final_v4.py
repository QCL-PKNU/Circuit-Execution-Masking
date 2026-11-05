#!/usr/bin/env python3
"""Hardware run v4: Q-MaskCut with the FULL evaluated method, 23 circuits (3 to 127 qubits).
Changes from hw_final.py (everything else identical, so it stays comparable with results/hw_final_v2):
  * key = random among the bit flips with the largest output TVD (circuits up to 14 qubits; random above)
  * H on every decoy at the end (decoys read 50/50, cannot be spotted)
  * RX(+t)/RX(-t) pair across every cut wire (identity; reconstruction exact)
  * coupling map without bad qubits/links (readout <= 5%, sx <= 1%, 2q <= 5%, largest connected part);
    the earlier map let a fragment land on a broken qubit in simulation
  * output folder results/hw_final_v4/<backend>; MQT bv_n3 / bv_n4 (data/circuits) added
Previous description:

Per circuit: the uncut ORIGINAL and every Q-MaskCut fragment instance, with the final setting:
key, 2 decoys, 2 CZ padlocks, CutQC cut (2-3 fragments up to 23 qubits; fragments <= 20 wires up
to 40 qubits and <= 27 above), each instance compiled at level 3 (broken couplers out, one
layout per fragment), then after compilation, on the qubits the circuit uses only: the smart
twirl (X around a two-qubit gate where the idle time holds it, virtual Z elsewhere) and
no-pattern DD (random-axis pulse pairs, at most MAX_PAIRS per idle gap). The original is compiled at
optimization level 3 and submitted as it is (no DD). Runtime DD and runtime gate/measurement twirling off (OPTS), 2048 shots.
Before submitting, every circuit is checked noiselessly: CutQC rebuild from exact (Aer) fragment
distributions, decoded with the key, against the ideal (whole output up to 29 protected wires,
both correct bitstrings beyond).

    .venv/bin/python scripts/hw_final.py --account pknu3 --backend ibm_marrakesh [--submit|--collect]
"""
from __future__ import annotations
import argparse, csv, json, math, pickle, statistics as st, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import new_workflow_test as W
W.BEST_OF = 8                      # same as the simulation runs (v4)
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qmaskcut import config as cfg
from qmaskcut.cutting import cutqc as C
from qmaskcut.masking import protect_general
from qmaskcut.metrics.ideal import ideal
from qmaskcut.metrics.normged import deterministic_normged as ng
from qmaskcut.metrics.scoring import score_dense, score, counts_to_le
from qmaskcut.utils.circuits import load
from hw_prepare import reference, qpu_estimate
from qiskit.circuit.library import HGate, RXGate
import itertools, random
from qmaskcut_eval import good_map

CIRCS = [("bv_n3", str(ROOT / "data/circuits/bv_n3.qasm"), 2), ("bv_n4", str(ROOT / "data/circuits/bv_n4.qasm"), 2),
         ("bell_n4", "small/bell_n4/bell_n4.qasm", 2), ("qpe_n9", "small/qpe_n9/qpe_n9.qasm", 2),
         ("adder_n10", "small/adder_n10/adder_n10.qasm", 2), ("seca_n11", "medium/seca_n11/seca_n11.qasm", 2),
         ("multiply_n13", "medium/multiply_n13/multiply_n13.qasm", 2), ("bv_n14", "medium/bv_n14/bv_n14.qasm", 2),
         ("qf21_n15", "medium/qf21_n15/qf21_n15.qasm", 2), ("qec9xz_n17", "medium/qec9xz_n17/qec9xz_n17.qasm", 2),
         ("bv_n19", "medium/bv_n19/bv_n19.qasm", 3), ("cat_state_n22", "medium/cat_state_n22/cat_state_n22.qasm", 3),
         ("ghz_state_n23", "medium/ghz_state_n23/ghz_state_n23.qasm", 3), ("knn_n25", "medium/knn_n25/knn_n25.qasm", None),
         ("ising_n26", "medium/ising_n26/ising_n26.qasm", None), ("ising_n34", "large/ising_n34/ising_n34.qasm", None),
         ("ising_n42", "large/ising_n42/ising_n42.qasm", None), ("ghz_n40", "large/ghz_n40/ghz_n40.qasm", None),
         ("dj_n50", "../MQTBench_qasm/dj_indep_qiskit_50.qasm", None), ("cat_n65", "large/cat_n65/cat_n65.qasm", None),
         ("dj_n70", "../MQTBench_qasm/dj_indep_qiskit_70.qasm", None), ("ghz_n78", "large/ghz_n78/ghz_n78.qasm", None),
         ("ghz_n127", "large/ghz_n127/ghz_n127.qasm", None)]
SHOTS, MAX_PAIRS, PADLOCKS, FULL_MAX, CHUNK = 2048, 8, 2, 29, 200
OPTS = {"dynamical_decoupling": {"enable": False},       # no IBM runtime DD and no IBM
        "twirling": {"enable_gates": False, "enable_measure": False}}   # runtime twirling for ANY circuit; the original runs
                                                                         # exactly as compiled; Q-MaskCut has its own masking


def counts(pub):
    data = pub.data
    reg = next(iter(data.__dict__)) if hasattr(data, "__dict__") else "meas"
    return getattr(data, reg).get_counts()


def exact_vectors(circs):
    sv = AerSimulator(method="statevector"); out = []
    for c in circs:
        g = QuantumCircuit(*c.qregs)
        for i in c.data:
            if i.operation.name not in ("measure", "barrier", "delay"):
                g.append(i.operation, i.qubits)
        # the measured qubits, in clbit order, give the probability vector CutQC expects
        m = {c.find_bit(i.clbits[0]).index: c.find_bit(i.qubits[0]).index for i in c.data if i.operation.name == "measure"}
        qs = [m[k] for k in range(len(m))]
        g = transpile(g, basis_gates=["cz", "rz", "sx", "x", "id", "u", "cx"], optimization_level=0)
        g.save_probabilities(qs)
        out.append(np.asarray(sv.run(g).result().data(0)["probabilities"], dtype=float))
    return out


WINDOW = 12   # spread-output circuits too wide for a statevector: scored on blocks of this many qubits


def window_reference(base, n, shots=200000):
    """No exact whole-output reference (wide, spread output, e.g. Ising, kNN): one exact MPS
    sampling, read as marginals on consecutive blocks of WINDOW original qubits."""
    from qiskit_aer import AerSimulator
    qc = base.copy(); qc.measure_all()
    sim = AerSimulator(method="matrix_product_state")
    qc = transpile(qc, basis_gates=["cx", "rz", "sx", "x", "h", "ry", "rx", "u"], optimization_level=0)
    cnt = sim.run(qc, shots=shots, seed_simulator=cfg.SEED).result().get_counts()
    cuts = list(range(0, n, WINDOW)); cuts = cuts[:-1] if n - cuts[-1] < WINDOW // 2 and len(cuts) > 1 else cuts
    wins = [list(range(a, cuts[i + 1] if i + 1 < len(cuts) else n)) for i, a in enumerate(cuts)]
    return {"windows": [(w, marginal({k.replace(" ", "")[::-1]: v / shots for k, v in cnt.items()}, w)) for w in wins]}


def marginal(p, w):
    out = {}
    for k, v in p.items():
        key = "".join(k[q] for q in w); out[key] = out.get(key, 0.0) + v
    return out


def hel(p, r):
    return sum(math.sqrt(max(p.get(k, 0.0), 0.0) * v) for k, v in r.items()) ** 2


def window_scores(cq, info, n, ref):
    f = []; nk = []
    for w, r in ref["windows"]:
        for out, inf in ((f, info), (nk, dict(info, flip="0" * n))):
            dv = C.window_decoded(cq, inf, w)
            out.append(hel({format(i, f"0{len(w)}b")[::-1]: float(x) for i, x in enumerate(dv)}, r))
    return st.mean(f), st.mean(nk)


def decoded(cq, info, n, ref, N):
    if isinstance(ref, dict) and "windows" in ref:
        return window_scores(cq, info, n, ref)
    if N <= FULL_MAX:
        return score_dense(C.full_decoded(cq, info, n)[0], ref)["fidelity"], \
               score_dense(C.full_decoded(cq, dict(info, flip="0" * n), n)[0], ref)["fidelity"]
    p = {k: max(C.point_decoded(cq, info, k), 0.0) for k in ref}
    pk = {k: max(C.point_decoded(cq, dict(info, flip="0" * n), k), 0.0) for k in ref}
    h = lambda q: sum(math.sqrt(q[k] * r) for k, r in ref.items()) ** 2
    return h(p), h(pk)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", required=True); ap.add_argument("--backend", default="ibm_marrakesh")
    ap.add_argument("--submit", action="store_true"); ap.add_argument("--collect", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--tag", default="", help="output sub-folder, so separate runs never overwrite each other")
    ap.add_argument("--max-qpu", type=float, default=None, help="submit only if the estimate is below this (s)")
    ap.add_argument("--submit-prepared", action="store_true",
                    help="submit the circuits already in <out>/prepared.pkl (same chip), without preparing again")
    a = ap.parse_args()
    out = ROOT / "results/hw_final_v4" / (a.backend + (f"_{a.tag}" if a.tag else "")); out.mkdir(parents=True, exist_ok=True)
    svc = QiskitRuntimeService(name=a.account); backend = svc.backend(a.backend); target = backend.target
    if a.collect:
        P = pickle.load(open(out / "prepared.pkl", "rb")); rec = json.loads((out / "jobs.json").read_text())
        res = {}
        for jid, lo, hi in rec["jobs"]:
            r = svc.job(jid).result()
            for k in range(hi - lo):
                res[lo + k] = counts(r[k])
        # raw IBM counts of every submitted circuit, so nothing depends on IBM keeping the jobs
        (out / "raw_counts.json").write_text(json.dumps({str(i): c for i, c in sorted(res.items())}))
        rows = []
        for g in P["groups"]:
            ref, n = g["ref"], g["n"]
            po = counts_to_le(res[g["orig"]], n)
            if "windows" in ref:
                f_orig = st.mean(hel(marginal(po, w), r) for w, r in ref["windows"])
            else:
                f_orig = sum(math.sqrt(po.get(k, 0) * r) for k, r in ref.items()) ** 2
            cq = C._finish(g["lab"], g["subs"], g["cpm"])
            vecs = [C.counts_to_probs(res[i], P["circuits"][i].num_clbits) for i in g["frag"]]
            C.evaluate_from(cq, dict(zip(g["labels"], vecs)))
            f, nk = decoded(cq, g["info"], n, ref, g["N"])
            row = {k: g[k] for k in ("circuit", "n", "K", "widths", "instances", "layers_original", "layers_fragment",
                                     "time_original_us", "time_fragment_us", "mask_pulses", "wires_masked",
                                     "normGED_fragment", "noiseless")} | \
                  {"fid_original": round(f_orig, 5), "fid_qmaskcut": round(f, 5), "fid_without_key": round(nk, 5)}
            rows.append(row); print(row, flush=True)
        with open(out / "results.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        for jid, _, _ in rec["jobs"]:
            try:
                print(jid, "QPU s:", svc.job(jid).usage())
            except Exception:
                pass
        sys.exit()
    gmap = good_map(target); region = len({q for e in gmap.get_edges() for q in e})
    circuits, groups = [], []
    if a.submit_prepared:
        P = pickle.load(open(out / "prepared.pkl", "rb")); circuits, groups = P["circuits"], P["groups"]
        a.submit = True
    for name, path, pieces in ([] if a.submit_prepared else CIRCS):
        if a.only and name not in a.only:
            continue
        t0 = time.time()
        base = load(name, path); n = base.num_qubits
        ref, how = reference(name, base)
        if ref is None:
            ref, how = window_reference(base, n), f"MPS marginals on {WINDOW}-qubit blocks"
        best_keys = None
        if n <= 14:                                           # best-TVD key from the exact output
            from qiskit.quantum_info import Statevector
            idl = Statevector(base).probabilities(); ft = {}
            for f in itertools.product([0, 1], repeat=n):
                if any(f):
                    kk = np.arange(2 ** n) ^ sum(b_ << q for q, b_ in enumerate(f)); ft["".join(map(str, f))] = 0.5 * np.abs(idl - idl[kk]).sum()
            best_keys = {x for x, y in ft.items() if y > max(ft.values()) - 1e-9}
        pick = None
        for sd in range(cfg.SEED, cfg.SEED + (300 if best_keys else 1)):
            prot, tags, info = protect_general(base, n_decoy=2, n_padlock=PADLOCKS, reps=2, seed=sd, n_twirl="random")
            v = ft.get(info["flip"], 0.0) if best_keys else 0.0
            if pick is None or v > pick[0] + 1e-12:
                pick = (v, sd, prot, tags, info)
            if not best_keys or info["flip"] in best_keys:
                break
        _, key_seed, prot, tags, info = pick
        lab = QuantumCircuit(*prot.qregs)
        for ins, t in zip(prot.data, tags):
            if t != "dd_pad":
                lab.append(ins.operation, ins.qubits)
        for d in info["decoy"]:                              # decoys read 50/50, like data
            lab.append(HGate(), [d])
        N = lab.num_qubits
        cmap = gmap if N <= region else W.working_map(target)
        if pieces:
            cq, _ = C.find_balanced_cut(lab, pieces, max_cuts=2, allow_extra_cut=False)
        else:
            cq, _ = C.cut_to_width(lab, 20 if n < 40 else 27)
        # cut check: share of the original's two-qubit gates in the largest fragment (padlocks are cz)
        orig2q = sum(1 for i in lab.data if len(i.qubits) == 2 and i.operation.name != "cz")
        share = max(sum(1 for i in sc.data if len(i.qubits) == 2 and i.operation.name != "cz")
                    for sc in cq.cut_solution["subcircuits"]) / max(orig2q, 1)
        if share > 0.999:
            print(f"WARNING {name}: one fragment holds the whole algorithm (100% of its two-qubit gates)", flush=True)
        rr = random.Random(1000 + key_seed); subs = [sc.copy() for sc in cq.cut_solution["subcircuits"]]
        pre = {i_: [] for i_ in range(len(subs))}
        for w_, path in C.cpm_index(cq).items():
            for (su, qu), (sdd, qd) in zip(path, path[1:]):
                t_ = rr.uniform(0, 2 * math.pi); subs[su].append(RXGate(t_), [qu]); pre[sdd].append((qd, -t_))
        for s_, items in pre.items():
            if items:
                h_ = QuantumCircuit(*subs[s_].qregs)
                for qd, t_ in items:
                    h_.append(RXGate(t_), [qd])
                subs[s_] = h_.compose(subs[s_])
        cq.cut_solution["subcircuits"] = subs                 # RX pair across every cut
        labels, frags = C.instance_circuits(cq)
        frags = [f if f.num_clbits else f.measure_all(inplace=False) for f in frags]
        b = base.copy(); b.measure_all(); c_orig = W.best(b, target, cmap); go = W.gates_only(c_orig)
        comp = [None] * len(frags)
        for s in sorted({x[0] for x in labels}):
            idx = [j for j, x in enumerate(labels) if x[0] == s]
            rep = W.best(frags[idx[0]], target, cmap)
            lay = rep.layout.initial_index_layout(filter_ancillas=True)[:frags[idx[0]].num_qubits]
            for j in idx:
                comp[j] = W.tp(frags[j], target, cmap, W.SEED, layout=lay)
        masked = [W.mask_fill(W.twirl_smart(c, target, cfg.SEED + j), target, cfg.SEED + j, max_pairs=MAX_PAIRS)
                  for j, c in enumerate(comp)]
        # the original is compiled at optimization level 3 and submitted as it is: twirl and DD belong to
        # Q-MaskCut only (no DD or twirl on the original)
        run_orig = c_orig
        # noiseless check from exact fragment distributions
        cq0 = C._finish(lab, cq.cut_solution["subcircuits"], cq.cut_solution["complete_path_map"])
        # each gap's masking pulses are checked to be the identity, then the rest is simulated exactly
        C.evaluate_from(cq0, dict(zip(labels, exact_vectors([W.strip_mask(c) for c in masked]))))
        f0, _ = decoded(cq0, info, n, ref, N)
        big = max(masked, key=lambda c: W.gates_only(c).depth())
        wm, wt = map(sum, zip(*[W.wires_masked(c) for c in masked]))
        g = {"circuit": name, "n": n, "N": N, "K": cq.cut_solution["num_cuts"], "widths": "+".join(map(str, C.widths(cq))),
             "instances": len(frags), "lab": lab, "subs": cq.cut_solution["subcircuits"],
             "cpm": cq.cut_solution["complete_path_map"], "labels": labels, "info": info, "ref": ref,
             "layers_original": go.depth(), "layers_fragment": W.gates_only(big).depth(),
             "time_original_us": round(run_orig.estimate_duration(target, unit="u"), 2),
             "time_fragment_us": round(big.estimate_duration(target, unit="u"), 2),
             "mask_pulses": sum(W.n_mask(c) for c in masked), "wires_masked": f"{wm}/{wt}",
             "normGED_fragment": round(ng(go, W.gates_only(big)), 4), "noiseless": round(f0, 4),
             "algorithm_share_largest_fragment": round(share, 3), "key": info["flip"], "key_seed": key_seed,
             "key_is_best": (info["flip"] in best_keys) if best_keys else None,
             "coupling": "good qubits" if N <= region else "all working qubits",
             "orig": len(circuits), "frag": list(range(len(circuits) + 1, len(circuits) + 1 + len(masked)))}
        circuits += [run_orig] + masked; groups.append(g)
        print(f"{name:14s} K={g['K']} {g['widths']:22s} inst {g['instances']:3d} | layers {g['layers_original']}->{g['layers_fragment']} "
              f"| time {g['time_original_us']}->{g['time_fragment_us']} us | pulses {g['mask_pulses']} wires {g['wires_masked']} "
              f"| nGED {g['normGED_fragment']} | noiseless {f0:.3f} | {time.time() - t0:.0f}s", flush=True)
    if not a.submit_prepared:
        pickle.dump({"circuits": circuits, "groups": groups}, open(out / "prepared.pkl", "wb"))
    est = qpu_estimate([W.gates_only(c).depth() for c in circuits], SHOTS)
    left = svc.usage().get("usage_remaining_seconds")
    print(f"{len(circuits)} circuits, estimated QPU {est:.0f} s, {left} s left on {a.account}")
    bad = [g["circuit"] for g in groups if g["noiseless"] < 0.95]
    if bad:
        print("NOISELESS CHECK FAILED, not submitting:", bad); sys.exit(1)
    if not a.submit:
        sys.exit()
    if est > left - 20 or (a.max_qpu is not None and est > a.max_qpu):
        print("not submitting: estimate", round(est), "s, limit", a.max_qpu, ", left", left); sys.exit(1)
    jobs = []
    for lo in range(0, len(circuits), CHUNK):
        hi = min(lo + CHUNK, len(circuits))
        j = SamplerV2(mode=backend, options=OPTS).run(circuits[lo:hi], shots=SHOTS)
        jobs.append((j.job_id(), lo, hi)); print("submitted", j.job_id(), lo, hi - 1, flush=True)
    (out / "jobs.json").write_text(json.dumps({"jobs": jobs, "account": a.account, "backend": a.backend, "shots": SHOTS,
                                               "options": OPTS, "max_pairs_per_gap": MAX_PAIRS, "padlocks": PADLOCKS,
                                               "submitted": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
