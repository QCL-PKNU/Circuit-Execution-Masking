#!/usr/bin/env python3
"""BV-3 under the pasted evaluation spec: original vs Q-MaskCut vs ObfusQate (inverse / composite / delayed gates)
vs ECQCO.

Spec: FakeMontrealV2 (noisy) + AerSimulator noiseless; 1024 shots; optimization_level 3 (all results use level 3); seed_transpiler 42;
20 iterations x 10 random U3 input states (one U3 per qubit before the circuit; input seed shared by every role of
an (iteration, input)); pieces compiled separately, never re-transpiled after recombination. Every circuit and count
file goes through RunStore (runs/<date>_bv3_spec/); metrics and figures come only from the saved files.

Roles: original (noisy), original_rerun (noise floor), compiler_view (+ per piece for Q-MaskCut), reconstructed
(noisy, after the key). TVD_ideal = exact noiseless reconstruction (statevector) vs the ideal output (< 0.01 check).

    .venv/bin/python trackb_bv3.py --run runs/bv3
"""
from __future__ import annotations
import argparse, itertools, json, math, os, random, sys, time
from pathlib import Path

os.environ.setdefault("QISKIT_PARALLEL", "FALSE")
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]
import numpy as np

SHOTS, OPT, SEED_T, ITERS, INPUTS = 1024, 3, 42, 20, 10   # level 3, as every other result
METHODS = ["original", "Q-MaskCut", "inverse gates", "composite gates", "delayed gates", "ECQCO"]


def tvd(a, b):
    return 0.5 * float(np.abs(np.asarray(a) - np.asarray(b)).sum())


def hell(a, b):
    return float(np.sum(np.sqrt(np.clip(a, 0, None) * np.asarray(b))) ** 2)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True)
    ap.add_argument("--iters", type=int, default=ITERS); ap.add_argument("--inputs", type=int, default=INPUTS)
    ap.add_argument("--it-start", type=int, default=0); ap.add_argument("--it-end", type=int, default=None)
    ap.add_argument("--ged-timeout", type=float, default=2.0)
    ap.add_argument("--key-mode", choices=["best", "uniform"], default="best",
                    help="best: search up to 300 seeds for a best-TVD key; uniform: the first draw (random key)")
    a = ap.parse_args()
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import HGate, RXGate
    from qiskit.quantum_info import Statevector, Operator
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime.fake_provider import FakeMontrealV2
    import new_workflow_test as W
    from qmaskcut.masking import protect_general
    from qmaskcut.cutting import cutqc as C
    from qmaskcut.utils.circuits import load, two_q_only, unmeasured
    from qmaskcut.metrics.normged import deterministic_normged as ng, normged_paper_eq14
    from qmaskcut.baselines.obfusqate import OBFUSQATE
    from qmaskcut_eval import circuit_path, good_map, ecqco_circuit
    from qexp_store import RunStore

    cfgd = {"benchmark": "bv_n3 (MQT, secret 11)", "backend_noisy": "FakeMontrealV2", "backend_ideal": "AerSimulator noiseless",
            "shots": SHOTS, "optimization_level": OPT, "seed_transpiler": SEED_T, "iterations": a.iters,
            "inputs": f"{a.inputs} random U3 states (one U3 per qubit), seed = 1000*iteration + input", "methods": METHODS,
            "q_maskcut": ("best-TVD key" if a.key_mode == "best" else "uniform random key") + " (seed = iteration), 2 decoys + 2 padlocks, H on decoys, CutQC 2 fragments, RX pair at cut, "
                         "fragments compiled separately (level 3), time-aware twirl + no-pattern DD",
            "coupling_map": "good qubits (readout<=5%, sx<=1%, 2q<=5%)"}
    st = RunStore(a.run, cfgd)
    log = open(Path(a.run) / f"log_it{a.it_start:02d}_{time.strftime('%Y%m%d_%H%M%S')}.txt", "w")
    say = lambda *x: (print(*x, flush=True), print(*x, file=log, flush=True))          # noqa: E731

    base0 = load("bv_n3", str(circuit_path("bv_n3"))); n = base0.num_qubits
    be = FakeMontrealV2(); target = be.target; cmap = good_map(target)
    noisy = AerSimulator.from_backend(be, method="statevector", max_parallel_shots=1)
    clean = AerSimulator(method="statevector", max_parallel_shots=1)
    basis = ["cx", "rz", "sx", "x", "id"]
    tp = lambda qc, layout=None: transpile(qc, basis_gates=basis, coupling_map=cmap, optimization_level=OPT,   # noqa: E731
                                           seed_transpiler=SEED_T, initial_layout=layout)
    n2 = lambda g: sum(1 for i in g.data if i.operation.num_qubits == 2)                  # noqa: E731
    dur = lambda t: round(t.estimate_duration(target, unit="u"), 3)                       # noqa: E731

    def run(B, label, method, role, qc, seed):
        # FAIR NOISE: non-Q-MaskCut circuits scheduled so idle time is charged (Q-MaskCut is scheduled by mask_fill)
        if B == "FakeMontrealV2" and method != "Q-MaskCut":
            qc = W.sched(qc, target); role = role + "_sched"
        if not st.has_counts(B, label, method, role):
            sim = noisy if B == "FakeMontrealV2" else clean
            c = sim.run(qc, shots=SHOTS, seed_simulator=seed).result().get_counts()
            st.save_circuit(B, label, method, role, qc); st.save_counts(B, label, method, role, c, SHOTS, seed_simulator=seed)
        rec = st.load_counts(B, label, method, role); assert rec["shots"] == SHOTS
        return rec["counts"]

    def marg(p_all, width):
        return p_all.reshape([2] * width).sum(axis=tuple(range(width - n))).reshape(-1) if width > n else p_all

    def overhead(logical, compiled):
        g = W.gates_only(compiled)
        free = transpile(logical, basis_gates=basis, optimization_level=OPT, seed_transpiler=SEED_T)
        return dict(depth=g.depth(), gates=g.size(), cx=n2(g), swaps=max(0, round((n2(g) - n2(free)) / 3)),
                    qubits=logical.num_qubits, time_us=dur(compiled))

    def lng(x, y):
        b = ng(x, y)
        try:
            v, secs, _ = normged_paper_eq14(x, y, timeout=a.ged_timeout)
        except RecursionError:
            return b, "bound"
        return (v, "exact") if secs < a.ged_timeout * 0.95 and v <= b + 1e-12 else (min(v, b), "bound")

    rows = []
    t0 = time.time()
    end = a.iters if a.it_end is None else a.it_end
    cases = [(it, k) for it in range(a.it_start, end) for k in range(a.inputs)] + ([(99, None)] if a.it_start == 0 else [])
    for it, k in cases:
            if k is None:
                qc = base0.copy(); label = "plain"; k = 0
            else:
                rng = np.random.default_rng(1000 * it + k)
                qc = QuantumCircuit(n)
                for q in range(n):
                    qc.u(*rng.uniform(0, 2 * np.pi, 3), q)
                qc.compose(base0, inplace=True)
                label = f"it{it:02d}_in{k:02d}"
            ideal = Statevector(qc).probabilities()
            m = qc.measure_all(inplace=False); co = tp(m)
            po = C.counts_to_probs(run("FakeMontrealV2", label, "original", "run", co, 10**6 * it + 10 * k + 1), n)
            pr = C.counts_to_probs(run("FakeMontrealV2", label, "original", "rerun", co, 10**6 * it + 10 * k + 2), n)
            floor = tvd(po, pr); cog = W.gates_only(co)
            common = dict(iteration=it, input=k, label=label, noise_floor=floor)
            ideal_bits = {format(i, f"0{n}b")[::-1]: float(x) for i, x in enumerate(ideal)}
            rows.append(dict(common, method="original", tvd_obf_min=tvd(po, ideal), tvd_deobf=tvd(po, ideal), tvd_ideal=0.0,
                             fidelity=hell(po, ideal), normged=0.0, normged_kind="exact", operator_check=True,
                             bits_cv=list(po), bits_rc=list(po), bits_ideal=list(ideal),
                             **overhead(qc, co), pieces=[tvd(po, ideal)]))
            # ObfusQate
            for j, (tech, make) in enumerate(OBFUSQATE.items()):
                random.seed(1000 * it + k)
                obf = unmeasured(make(qc.copy())); cs = tp(obf.measure_all(inplace=False))
                p = marg(C.counts_to_probs(run("FakeMontrealV2", label, tech, "compiler_view", cs, 10**6 * it + 10 * k + 3 + j),
                                           obf.num_qubits), obf.num_qubits)
                ex = marg(Statevector(obf).probabilities(), obf.num_qubits)
                v, kind = lng(qc, obf)
                rows.append(dict(common, method=tech, bits_cv=list(p), bits_rc=list(p), tvd_obf_min=tvd(p, ideal), tvd_deobf=tvd(p, ideal), tvd_ideal=tvd(ex, ideal),
                                 fidelity=hell(p, ideal), normged=v, normged_kind=kind,
                                 operator_check=bool(Operator(qc).equiv(Operator(obf))) if obf.num_qubits == n else None,
                                 **overhead(obf, cs), pieces=[tvd(p, ideal)]))
            # ECQCO
            try:
                e = ecqco_circuit(f"bv_n3_spec_{label}", qc)
            except Exception:                                    # noqa: BLE001
                e = None
            if e is None:
                rows.append(dict(common, method="ECQCO", status="N/A"))
            else:
                obf = unmeasured(e); cs = tp(obf.measure_all(inplace=False))
                p = marg(C.counts_to_probs(run("FakeMontrealV2", label, "ECQCO", "compiler_view", cs, 10**6 * it + 10 * k + 6),
                                           obf.num_qubits), obf.num_qubits)
                v, kind = lng(qc, obf)
                rows.append(dict(common, method="ECQCO", bits_cv=list(p), bits_rc=list(p), tvd_obf_min=tvd(p, ideal), tvd_deobf=tvd(p, ideal),
                                 tvd_ideal=tvd(marg(Statevector(obf).probabilities(), obf.num_qubits), ideal), fidelity=hell(p, ideal),
                                 normged=v, normged_kind=kind, operator_check=None, **overhead(obf, cs), pieces=[tvd(p, ideal)],
                                 note="ECQCO's key does not recover the output (decodable = False)"))
            # Q-MaskCut
            flip_tvd = {}
            for f in itertools.product([0, 1], repeat=n):
                if any(f):
                    kk = np.arange(2 ** n) ^ sum(b << q for q, b in enumerate(f)); flip_tvd["".join(map(str, f))] = tvd(ideal, ideal[kk])
            BEST = {x for x, y in flip_tvd.items() if y > max(flip_tvd.values()) - 1e-9}
            for sd in range(100 * it, 100 * it + (300 if a.key_mode == "best" else 1)):
                prot, tags, info = protect_general(qc, n_decoy=2, n_padlock=2, reps=2, seed=sd, n_twirl="random")
                if a.key_mode == "uniform" or info["flip"] in BEST:
                    break
            lab = QuantumCircuit(*prot.qregs)
            for ins, tg in zip(prot.data, tags):
                if tg != "dd_pad":
                    lab.append(ins.operation, ins.qubits)
            for d in info["decoy"]:
                lab.append(HGate(), [d])
            cq, _ = C.find_balanced_cut(lab, 2, max_cuts=2, allow_extra_cut=True)
            rr = random.Random(sd); subs = [s.copy() for s in cq.cut_solution["subcircuits"]]; pre = {i: [] for i in range(len(subs))}
            for w, path in C.cpm_index(cq).items():
                for (su, qu), (sdd, qd) in zip(path, path[1:]):
                    t = rr.uniform(0, 2 * math.pi); subs[su].append(RXGate(t), [qu]); pre[sdd].append((qd, -t))
            for s_, items in pre.items():
                if items:
                    h = QuantumCircuit(*subs[s_].qregs)
                    for qd, t in items:
                        h.append(RXGate(t), [qd])
                    subs[s_] = h.compose(subs[s_])
            cq.cut_solution["subcircuits"] = subs
            labels, circs = C.instance_circuits(cq)
            jobs = [None] * len(circs)
            for fr in sorted({l[0] for l in labels}):           # pieces compiled separately, one layout per piece
                idx = [j for j, l in enumerate(labels) if l[0] == fr]
                rep = tp(circs[idx[0]].measure_all(inplace=False))
                lay = rep.layout.initial_index_layout(filter_ancillas=True)[:circs[idx[0]].num_qubits]
                for j in idx:
                    t = tp(circs[j].measure_all(inplace=False), layout=lay)
                    t = W.mask_fill(W.twirl_smart(t, target, sd + j), target, sd + j, max_pairs=8)
                    jobs[j] = t
            vec = {l: C.counts_to_probs(run("FakeMontrealV2", label, "Q-MaskCut", f"inst_{j}", jobs[j], 10**6 * it + 1000 + 10 * k + j),
                                        circs[j].num_qubits) for j, l in enumerate(labels)}
            C.evaluate_from(cq, vec)
            dec, _, _ = C.full_decoded(cq, info, n); nok, _, _ = C.full_decoded(cq, dict(info, flip="0" * n), n)
            C.evaluate_from(cq, {l: Statevector(c).probabilities() for l, c in zip(labels, circs)})
            exd, _, _ = C.full_decoded(cq, info, n)
            # TVD_obf per piece: each piece's data qubits (all-zero init, computational meas), no key, vs ideal marginal
            cpm = C.cpm_index(cq); real = info["real"]; pieces = []
            for fr in sorted({l[0] for l in labels}):
                jj = [j for j, l in enumerate(labels) if l[0] == fr and all(x == "zero" for x in l[1]) and all(x == "comp" for x in l[2])]
                ends = {qq: cpm[real[qq]][-1][1] for qq in range(n) if cpm[real[qq]][-1][0] == fr}
                if not jj or not ends:
                    continue
                w = circs[jj[0]].num_qubits; pf = vec[labels[jj[0]]].reshape([2] * w); fq = list(ends)
                keep = [w - 1 - ends[qq] for qq in fq]
                mf = np.transpose(pf.sum(axis=tuple(x for x in range(w) if x not in keep)), [sorted(keep).index(x) for x in keep])
                ax = sorted([n - 1 - qq for qq in fq])
                mi = np.transpose(ideal.reshape([2] * n).sum(axis=tuple(x for x in range(n) if (n - 1 - x) not in fq)),
                                  [ax.index(n - 1 - qq) for qq in fq])
                pieces.append(tvd(mf.reshape(-1), mi.reshape(-1)))
            v, kind = lng(qc, lab)
            big = max(jobs, key=lambda t: W.gates_only(t).depth())
            ov = overhead(subs[0], big)
            rows.append(dict(common, method="Q-MaskCut", bits_cv=list(nok), bits_rc=list(dec), tvd_obf_min=min(pieces) if pieces else tvd(nok, ideal), tvd_obf_whole=tvd(nok, ideal),
                             tvd_deobf=tvd(dec, ideal), tvd_ideal=tvd(exd, ideal), fidelity=hell(dec, ideal), normged=v, normged_kind=kind,
                             operator_check="n/a (classical reconstruction; exact check = TVD_ideal)", key=info["flip"], **ov, pieces=pieces))
            if k == 0:
                say(f"iteration {it:2d} done ({time.time() - t0:.0f}s)")
            if it == 0 and k == 0:                               # keep circuits for fig C
                st.save_circuit("logical", label, "Q-MaskCut", "protected", lab) if not st.has_circuit("logical", label, "Q-MaskCut", "protected") else None
                for s_, sub in enumerate(subs):
                    if not st.has_circuit("logical", label, "Q-MaskCut", f"piece_{s_}"):
                        st.save_circuit("logical", label, "Q-MaskCut", f"piece_{s_}", sub)
    (Path(a.run) / f"rows_{a.it_start:02d}_{end:02d}.json").write_text(json.dumps(rows, indent=0, default=str))
    say(f"done: {len(rows)} rows in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
