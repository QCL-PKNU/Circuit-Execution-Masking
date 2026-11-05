#!/usr/bin/env python3
"""Tier-1 evaluation (full simulation, circuits up to MAX_DENSE qubits): Q-MaskCut vs existing methods.
Every circuit and every count file is saved once with RunStore (qexp_store.py) and reused; metrics and
figures are computed only from the saved files (eval_figures.py).

Methods (same compiler: level 3, best of --best-of compiles by estimated success, on a coupling map without
bad qubits/links; same simulator, noise and shots):
  original                               run as is (no DD); rerun with another seed = noise floor
  inverse / composite / delayed gates    OBFUSQATE baselines, run as is; compiler view = output
  ECQCO                                  authors' code (where it runs); its key cannot recover the output
  Q-MaskCut                              best-TVD key, 2 decoys + 2 padlocks, H on decoys, CutQC cut,
                                         RX pair at the cut, one layout per fragment, no-pattern DD

Roles saved per (backend, circuit, method): circuits/<b>/<c>/<method>__<role>.qasm and
counts/<b>/<c>/<method>__<role>.json; Q-MaskCut saves one role per fragment instance (inst_<j>).

    .venv/bin/python qmaskcut_eval.py --run runs/tier1_tier2 --circuits bv_n3 bv_n4 bell_n4 --noise none
    .venv/bin/python qmaskcut_eval.py --run runs/tier1_tier2 --set tier1 --noise FakeMontrealV2
"""
from __future__ import annotations
import argparse, glob, itertools, json, logging, math, os, random, subprocess, sys, time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("QISKIT_PARALLEL", "FALSE")
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]
MAX_DENSE = 22          # full output vectors up to this many qubits (2^22 floats)
SETS = {
    "small": ["bv_n3", "bv_n4", "bell_n4"],
    "tier1": ["bv_n3", "bv_n4", "bell_n4", "qpe_n9", "adder_n10", "seca_n11", "multiply_n13", "bv_n14",
              "qf21_n15", "qec9xz_n17", "bv_n19"],
}
A = None
ALL_METHODS = ["original", "inverse gates", "composite gates", "delayed gates", "ECQCO", "Q-MaskCut"]


DEFAULT_METHODS = list(ALL_METHODS)


class _Skip(Exception):
    pass


class _TooDeep(Exception):
    pass


def circuit_path(name):
    """data/circuits (our MQT circuits) first, then QASMBench, then MQT-Bench dj files."""
    p = ROOT / "data/circuits" / f"{name}.qasm"
    if p.exists():
        return p
    hits = glob.glob(str(ROOT / f"QASMBench/*/{name}/{name}.qasm"))
    if hits:
        return Path(hits[0])
    fam, size = name.rsplit("_n", 1)
    hits = glob.glob(str(ROOT / f"MQTBench_*/{fam}_indep_qiskit_{size}.qasm"))
    if hits:
        return Path(hits[0])
    raise FileNotFoundError(name)


def tvd(a, b):
    import numpy as np
    return 0.5 * float(np.abs(np.asarray(a) - np.asarray(b)).sum())


def good_map(target, ro=0.05, q1=0.01, q2=0.05):
    """Coupling map without bad qubits/links (readout > ro, sx > q1, 2-qubit gate > q2), largest connected part."""
    import rustworkx as rx
    from qiskit.transpiler import CouplingMap
    import new_workflow_test as W
    g2 = W.two_q(target)
    ok = {q for q in range(target.num_qubits)
          if (target["measure"][(q,)].error or 0) <= ro and (target["sx"][(q,)].error or 0) <= q1}
    edges = [qs for qs, pr in target[g2].items() if qs[0] in ok and qs[1] in ok and pr is not None and (pr.error or 0) <= q2]
    keep = max(rx.connected_components(CouplingMap(edges).graph.to_undirected()), key=len)
    return CouplingMap([e for e in edges if e[0] in keep and e[1] in keep])


def ecqco_circuit(name, base):
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, qasm2
    from qmaskcut import config as cfg
    d = ROOT / "results/qasm_small_v2/ecqco_circuits"; d.mkdir(parents=True, exist_ok=True)   # shared cache
    out, fail = d / name, d / f"{name}.failed"
    if out.exists():
        return QuantumCircuit.from_qasm_file(str(out))
    if fail.exists() or not cfg.ECQCO_PY.exists():
        return None
    n = base.num_qubits
    qr, cr = QuantumRegister(n, "q"), ClassicalRegister(n, "c")
    src = QuantumCircuit(qr, cr); src.compose(base, qubits=range(n), inplace=True); src.measure(qr, cr)
    (d / f"{name}.qasm").write_text(qasm2.dumps(src))
    (d / f"list_{name}.json").write_text(json.dumps([{"name": name, "qasm": str(d / f"{name}.qasm"), "qubits": n}]))
    try:
        p = subprocess.run([str(cfg.ECQCO_PY), str(cfg.ECQCO_WRAPPERS / "run_ecqco_any.py"), "--list",
                            str(d / f"list_{name}.json"), "--out", str(d), "--seed", str(cfg.SEED), "--lamb", "false"],
                           capture_output=True, text=True, timeout=900)
        msg = (p.stdout + p.stderr)[-2000:]
    except subprocess.TimeoutExpired:
        msg = "ECQCO's code did not finish within 900 s"
    if not out.exists():
        fail.write_text(msg)
        return None
    return QuantumCircuit.from_qasm_file(str(out))


def _init(args):
    global A
    A = args


def one(name):
    """Simulate (or reuse) every role of every method for one circuit on one backend; return metric rows."""
    import numpy as np
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import HGate, RXGate
    from qiskit.quantum_info import Statevector, Operator
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime import fake_provider
    import new_workflow_test as W
    from qmaskcut.masking import protect_general
    from qmaskcut.cutting import cutqc as C
    from qmaskcut.utils.circuits import load, two_q_only, unmeasured
    from qmaskcut.metrics.normged import deterministic_normged as ng, normged_paper_eq14
    from qmaskcut.baselines.obfusqate import OBFUSQATE
    from qmaskcut import config as cfg
    from qexp_store import RunStore

    W.BEST_OF = A.best_of
    want = lambda m: m in A.methods                                           # noqa: E731
    st = RunStore(A.run)
    B = A.noise
    t0 = time.time(); rows = []
    try:
        base = load(name, str(circuit_path(name))); n = base.num_qubits
        info_row = {"circuit": name, "backend": B, "n": n, "gates": base.size(),
                    "two_q": sum(1 for i in base.data if i.operation.num_qubits == 2)}
        if n > MAX_DENSE:
            return [dict(info_row, method="-", status=f"skipped: {n} qubits > {MAX_DENSE} (tier 2)")]
        ideal = Statevector(base).probabilities()                              # bit j = qubit j
        fid = lambda d: float(np.sum(np.sqrt(np.clip(d, 0, None) * ideal)) ** 2)   # noqa: E731

        flip_tvd = {}
        if n <= 14:
            for f in itertools.product([0, 1], repeat=n):
                if any(f):
                    k = np.arange(2 ** n) ^ sum(b << j for j, b in enumerate(f))
                    flip_tvd["".join(map(str, f))] = tvd(ideal, ideal[k])
        BEST = sorted(k for k, v in flip_tvd.items() if v > max(flip_tvd.values()) - 1e-9) if flip_tvd else []
        info_row["max_tvd"] = max(flip_tvd.values()) if flip_tvd else None

        backend = getattr(fake_provider, B if B != "none" else "FakeMarrakesh")()
        target = backend.target; cmap = good_map(target)
        opts = dict(method="statevector", max_parallel_shots=1, max_parallel_threads=A.threads)   # parallel over circuits instead
        sim = AerSimulator(**opts) if B == "none" else AerSimulator.from_backend(backend, **opts)
        dur = lambda t: round(t.estimate_duration(target, unit="u"), 3)                          # noqa: E731
        n2 = lambda g: sum(1 for i in g.data if i.operation.num_qubits == 2)                     # noqa: E731
        basis = [W.two_q(target), "rz", "sx", "x", "id"]

        def swaps(logical, compiled_gates):
            """estimated SWAPs = (2q gates after routing - 2q gates without routing) / 3"""
            free = transpile(logical, basis_gates=basis, optimization_level=3, seed_transpiler=W.SEED)
            return max(0, round((n2(compiled_gates) - n2(free)) / 3))

        def counts_for(method, role, compiled, seed):
            # Fair noise: every non-Q-MaskCut circuit is scheduled (idle time as explicit delays) so the noise
            # model charges idle decoherence to it too; Q-MaskCut is scheduled already by mask_fill. Unscheduled originals
            # (the v4 first pass) got no idle noise and looked far too good (qpe_n9 0.80 vs 0.51 scheduled).
            if B != "none" and method != "Q-MaskCut" and A.fair_sched:
                compiled = W.sched(compiled, target); role = role + "_sched"
            if not st.has_counts(B, name, method, role):
                act = len({compiled.find_bit(q).index for i in compiled.data if i.operation.name not in ("barrier", "delay")
                           for q in i.qubits})
                cost = compiled.size() * 2.0 ** act
                if B != "none" and cost > A.max_cost:
                    raise _TooDeep(f"not simulated: too deep for noisy simulation ({compiled.size()} gates x 2^{act} "
                                   f"= {cost:.1e} per shot > {A.max_cost:.0e})")
                c = sim.run(compiled, shots=A.shots, seed_simulator=seed).result().get_counts()
                st.save_counts(B, name, method, role, c, A.shots, seed_simulator=seed)
                try:
                    st.save_circuit(B, name, method, role, compiled)
                except FileExistsError:                       # saved already by a parallel job: keep the first copy
                    pass
            return st.load_counts(B, name, method, role)["counts"]

        def probs(counts, width):
            return C.counts_to_probs(counts, width)

        def logical_ng(a, b):
            """networkx exact GED only for small graphs (it recurses per node); otherwise the deterministic bound."""
            bound = ng(a, b)
            if a.size() + b.size() > A.ged_exact_max:
                return bound, "bound"
            try:
                v, secs, _ = normged_paper_eq14(a, b, timeout=A.ged_timeout)
            except RecursionError:
                return bound, "bound"
            exact = secs < A.ged_timeout * 0.95
            return (v, "exact") if exact and v <= bound + 1e-12 else (min(v, bound), "bound")

        def top(vec, k=64):
            idx = np.argsort(-np.abs(vec))[:k]
            return {format(int(i), f"0{n}b")[::-1]: float(vec[i]) for i in idx if abs(vec[i]) > 1e-9}

        seedb = lambda role_id: A.seed * 10**7 + role_id * 10**5 + 1                              # noqa: E731

        # ---- original + rerun (noise floor)
        b = base.copy(); b.measure_all(); co = W.best(b, target, cmap); cog = W.gates_only(co)
        po = probs(counts_for("original", "run", co, seedb(1)), n)
        pr = probs(counts_for("original", "rerun", co, seedb(2)), n)
        noise_floor = tvd(po, pr)
        row0 = dict(info_row, method="original", fidelity=fid(po), tvd_deobf=tvd(po, ideal), tvd_obf=tvd(po, ideal),
                    tvd_obf_min=tvd(po, ideal), tvd_ideal=0.0, noise_floor=noise_floor, normged_logical=0.0,
                    normged_logical_kind="exact", normged=0.0, ged2q=0.0, depth=cog.depth(), gates_compiled=cog.size(),
                    two_q_compiled=n2(cog), swaps=swaps(base, cog), qubits=n, time_us=dur(co), operator_check=True,
                    status="ok", bits={"ideal": top(ideal), "original": top(po)})
        if want("original"):
            rows.append(row0)

        def g2(a, bb):
            ta, tb = two_q_only(a), two_q_only(bb)
            return 0.0 if ta.size() == 0 and tb.size() == 0 else ng(ta, tb)

        # ---- baselines
        schemes = []
        for tech, make in OBFUSQATE.items():
            if not want(tech):
                continue
            random.seed(cfg.SEED)
            try:
                schemes.append((tech, unmeasured(make(base.copy()))))
            except Exception as e:                                   # noqa: BLE001
                rows.append(dict(info_row, method=tech, status=f"FAIL {type(e).__name__}: {str(e)[:100]}"))
        skip_e = A.no_ecqco or not want("ECQCO")
        e = None if skip_e else ecqco_circuit(name, base)
        if e is not None:
            schemes.append(("ECQCO", unmeasured(e)))
        elif not skip_e:
            rows.append(dict(info_row, method="ECQCO", status="N/A (ECQCO's code fails on this circuit)"))
        for k, (tech, obf) in enumerate(schemes):
            try:
                o = obf.copy(); o.measure_all(); cs = W.best(o, target, cmap); csg = W.gates_only(cs)
                pa = probs(counts_for(tech, "compiler_view", cs, seedb(10 + k)), obf.num_qubits)
                p = pa.reshape([2] * obf.num_qubits).sum(axis=tuple(range(obf.num_qubits - n))).reshape(-1) \
                    if obf.num_qubits > n else pa
                exact = Statevector(obf).probabilities()
                ex = exact.reshape([2] * obf.num_qubits).sum(axis=tuple(range(obf.num_qubits - n))).reshape(-1) \
                    if obf.num_qubits > n else exact
                opc = bool(Operator(base).equiv(Operator(obf))) if obf.num_qubits == n and n <= 10 else None
                lv, lk = logical_ng(base, obf)
                rows.append(dict(info_row, method=tech, fidelity=fid(p), tvd_deobf=tvd(p, ideal), tvd_obf=tvd(p, ideal),
                                 tvd_obf_min=tvd(p, ideal), tvd_ideal=tvd(ex, ideal), noise_floor=noise_floor,
                                 normged_logical=lv, normged_logical_kind=lk, normged=ng(cog, csg), ged2q=g2(cog, csg),
                                 depth=csg.depth(), gates_compiled=csg.size(), two_q_compiled=n2(csg),
                                 swaps=swaps(obf, csg), qubits=obf.num_qubits, time_us=dur(cs), operator_check=opc,
                                 decodable=(tech != "ECQCO"), status="ok",
                                 bits={"compiler_view": top(p), "reconstructed": top(p)}))
            except _TooDeep as td:
                rows.append(dict(info_row, method=tech, status=str(td)))
            except Exception as ex_:                                 # noqa: BLE001
                rows.append(dict(info_row, method=tech, status=f"FAIL {type(ex_).__name__}: {str(ex_)[:100]}"))

        # ---- Q-MaskCut
        if not want("Q-MaskCut"):
            raise _Skip

        def mask(t, s):
            """the paper's after-compile masking (smart8): time-aware Pauli twirl around every 2-qubit gate
            (X only where idle time holds it, else virtual Z; no extra time), then no-pattern DD in every idle
            gap of the active qubits. strip_mask checks every DD gap multiplies to the identity."""
            if A.twirl:
                t = W.twirl_smart(t, target, s)
            if A.dd:
                t = W.mask_fill(t, target, s, max_pairs=A.dd_pairs); W.strip_mask(t)
            return t

        def protect(seed):
            prot, tags, inf = protect_general(base, n_decoy=A.decoys, n_padlock=A.padlocks, reps=2, seed=seed, n_twirl="random")
            lab = QuantumCircuit(*prot.qregs)
            for ins, tg in zip(prot.data, tags):
                if tg != "dd_pad":
                    lab.append(ins.operation, ins.qubits)
            for d in inf["decoy"]:
                lab.append(HGate(), [d])
            return lab, inf

        pick = None
        for seed in range(A.seed, A.seed + (A.key_search if BEST and A.key_mode == "best" else 1)):
            lab, info = protect(seed)
            v = flip_tvd.get(info["flip"], 0.0)
            if pick is None or v > pick[0] + 1e-12:
                pick = (v, seed, lab, info)
            if not BEST or A.key_mode == "uniform" or info["flip"] in BEST:
                break
        _, seed, lab, info = pick
        N = lab.num_qubits
        st.save_circuit(B, name, "Q-MaskCut", "protected_logical", lab) if not st.has_circuit(B, name, "Q-MaskCut", "protected_logical") else None
        m = lab.copy(); m.measure_all()
        whole = mask(W.best(m, target, cmap), seed); wg = W.gates_only(whole)
        lv, lk = logical_ng(base, lab)
        n_lab = lambda t, lb: sum(1 for i in t.data if getattr(i.operation, "label", None) == lb)   # noqa: E731
        q = dict(info_row, method="Q-MaskCut", twirl_gates_whole=n_lab(whole, "twirl"), dd_gates_whole=n_lab(whole, "mask"), key=info["flip"], key_is_best=(info["flip"] in BEST) if BEST else None,
                 key_seed=seed, normged_logical=lv, normged_logical_kind=lk, normged=ng(cog, wg), ged2q=g2(cog, wg),
                 qubits=N, decodable=True)
        pieces = []
        if len(C.independent_pieces(lab)) > 1:                       # already in pieces: run whole
            v = probs(counts_for("Q-MaskCut", "whole", whole, seedb(30)), N)
            wires = list(reversed(range(N)))
            dec = C.decode_dense(v, wires, info, list(range(n)))
            nok = C.decode_dense(v, wires, dict(info, flip="0" * n), list(range(n)))
            exact_dec = C.decode_dense(Statevector(lab).probabilities(), wires, info, list(range(n)))
            q.update(cuts=0, widths=str(N), instances=1, depth=wg.depth(), gates_compiled=wg.size(),
                     two_q_compiled=n2(wg), swaps=swaps(lab, wg), time_us=dur(whole), note="not connected: run uncut")
            pieces.append(tvd(nok, ideal))
        else:
            cq, _ = C.find_balanced_cut(lab, 2, max_cuts=2, allow_extra_cut=True)
            r = random.Random(1000 + seed); subs = [s.copy() for s in cq.cut_solution["subcircuits"]]
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
            cq.cut_solution["subcircuits"] = subs
            for s, sub in enumerate(subs):
                if not st.has_circuit(B, name, "Q-MaskCut", f"fragment_{s}"):
                    st.save_circuit(B, name, "Q-MaskCut", f"fragment_{s}", sub)
            labels, circs = C.instance_circuits(cq)
            meas = [c.copy() for c in circs]
            for c in meas:
                c.measure_all()
            jobs = [None] * len(circs)
            for s in sorted({l[0] for l in labels}):                 # one layout per fragment
                idx = [j for j, l in enumerate(labels) if l[0] == s]
                rep = W.best(meas[idx[0]], target, cmap)
                lay = rep.layout.initial_index_layout(filter_ancillas=True)[:circs[idx[0]].num_qubits]
                for j in idx:
                    jobs[j] = mask(W.tp(meas[j], target, cmap, W.SEED, layout=lay), seed + j)
            vec = {}
            for j, (l, c) in enumerate(zip(labels, circs)):
                vec[l] = probs(counts_for("Q-MaskCut", f"inst_{j}__f{l[0]}_{'-'.join(l[1])}_{'-'.join(l[2])}",
                                          jobs[j], seedb(40 + j)), c.num_qubits)
            C.evaluate_from(cq, vec)
            dec, _, _ = C.full_decoded(cq, info, n)
            nok, _, _ = C.full_decoded(cq, dict(info, flip="0" * n), n)
            # TVD_ideal: the same cut, exact fragment outputs (no shots, no noise)
            C.evaluate_from(cq, {l: Statevector(c).probabilities() for l, c in zip(labels, circs)})
            exact_dec, _, _ = C.full_decoded(cq, info, n)
            # decoded output and exact output, for TVD_obf averaged over keys (key = XOR, so view under key k = dec[x ^ k])
            dist = Path(A.run) / "dists" / B; dist.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dist / f"{name}.npz", dec=np.asarray(dec), ideal=np.asarray(ideal),
                                exact_dec=np.asarray(exact_dec), nok=np.asarray(nok), key=info["flip"])
            # TVD_obf per piece: one cloud's fragment (all-zero init, computational meas), its data qubits,
            # read WITHOUT the key, vs the original's marginal on the same qubits (attacker told the wire map)
            cpm = C.cpm_index(cq); real = info["real"]
            for s in sorted({l[0] for l in labels}):
                j = next(i for i, l in enumerate(labels) if l[0] == s and all(x == "zero" for x in l[1])
                         and all(x in ("comp",) for x in l[2])) if any(
                    l[0] == s and all(x == "zero" for x in l[1]) and all(x == "comp" for x in l[2]) for l in labels) else None
                if j is None:
                    continue
                ends = {qd: path[-1][1] for qd, path in ((qq, cpm[real[qq]]) for qq in range(n)) if path[-1][0] == s}
                if not ends:
                    continue
                fq = list(ends); pf = vec[labels[j]]; w = circs[j].num_qubits
                pf_t = pf.reshape([2] * w)
                keep_axes = [w - 1 - ends[qq] for qq in fq]
                mf = pf_t.sum(axis=tuple(a for a in range(w) if a not in keep_axes))
                mf = np.transpose(mf, [sorted(keep_axes).index(a) for a in keep_axes])     # axis i -> qubit fq[i]
                it = ideal.reshape([2] * n)
                mi = it.sum(axis=tuple(a for a in range(n) if (n - 1 - a) not in fq))
                ax_i = sorted([n - 1 - qq for qq in fq])
                mi = np.transpose(mi, [ax_i.index(n - 1 - qq) for qq in fq])
                pieces.append(tvd(mf.reshape(-1), mi.reshape(-1)))
            big = max(jobs, key=lambda t: W.gates_only(t).depth()); bg = W.gates_only(big)
            q.update(twirl_gates_fragments=sum(n_lab(t, "twirl") for t in jobs), dd_gates_fragments=sum(n_lab(t, "mask") for t in jobs))
            q.update(cuts=cq.cut_solution["num_cuts"], widths="+".join(map(str, C.widths(cq))), instances=len(circs),
                     depth=bg.depth(), gates_compiled=bg.size(), two_q_compiled=n2(bg),
                     swaps=max(swaps(c, W.gates_only(t)) for c, t in zip(circs, jobs)),
                     time_us=max(dur(t) for t in jobs))
        q.update(fidelity=fid(dec), tvd_deobf=tvd(dec, ideal), tvd_obf=tvd(nok, ideal),
                 tvd_obf_pieces=pieces, tvd_obf_min=min(pieces) if pieces else tvd(nok, ideal),
                 tvd_ideal=tvd(exact_dec, ideal), noise_floor=noise_floor,
                 operator_check="n/a: classical reconstruction (exact check: TVD_ideal)",
                 status="ok", bits={"compiler_view": top(nok), "reconstructed": top(dec)})
        rows.append(q)
    except _Skip:
        pass
    except Exception as e:                                           # noqa: BLE001
        import traceback
        rows.append({"circuit": name, "backend": B, "method": "Q-MaskCut",
                     "status": f"FAIL {type(e).__name__}: {str(e)[:200]}", "trace": traceback.format_exc()[-800:]})
    for r in rows:
        r["seconds"] = round(time.time() - t0, 1)
    tag = "" if sorted(A.methods) == sorted(DEFAULT_METHODS) else "__" + "-".join(m.replace(" ", "") for m in A.methods)
    st.save_metrics(B, name + tag, rows)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="RunStore folder, e.g. runs/tier1_tier2")
    ap.add_argument("--circuits", nargs="*"); ap.add_argument("--set", choices=list(SETS))
    ap.add_argument("--noise", choices=["none", "FakeMarrakesh", "FakeMontrealV2"], default="none")
    ap.add_argument("--shots", type=int, default=8192); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--best-of", type=int, default=8); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--decoys", type=int, default=2); ap.add_argument("--padlocks", type=int, default=2)
    ap.add_argument("--key-search", type=int, default=300); ap.add_argument("--dd-pairs", type=int, default=8)
    ap.add_argument("--key-mode", choices=["best", "uniform"], default="best",
                    help="best: search up to --key-search seeds for a best-TVD key; uniform: the first draw (random key)")
    ap.add_argument("--ged-timeout", type=float, default=20.0); ap.add_argument("--no-ecqco", action="store_true")
    ap.add_argument("--methods", nargs="*", default=DEFAULT_METHODS,
                    help=f"subset of {ALL_METHODS}; results of a subset go to metrics/<backend>/<circuit>__<methods>.json")
    ap.add_argument("--max-cost", type=float, default=2e9, help="skip a noisy run when gates x 2^qubits per shot exceeds this")
    ap.add_argument("--no-fair-sched", dest="fair_sched", action="store_false",
                    help="do NOT schedule originals/baselines (the unfair first-pass setting)")
    ap.add_argument("--threads", type=int, default=1, help="Aer threads per worker process")
    ap.add_argument("--ged-exact-max", type=int, default=60, help="networkx exact GED only up to this many gates in total")
    ap.add_argument("--no-twirl", dest="twirl", action="store_false", help="skip the time-aware Pauli twirl")
    ap.add_argument("--no-dd", dest="dd", action="store_false", help="skip the no-pattern DD")
    args = ap.parse_args()
    names = args.circuits or SETS[args.set or "small"]
    config = {"shots": args.shots, "best_of": args.best_of, "seed": args.seed, "decoys": args.decoys,
              "padlocks": args.padlocks, "key_mode": args.key_mode, "key_search": args.key_search if args.key_mode == "best" else 1, "dd_pairs": args.dd_pairs,
              "optimization_level": 3, "coupling_map": "good qubits (readout<=5%, sx<=1%, 2q<=5%)",
              "methods": ["original", "inverse gates", "composite gates", "delayed gates", "ECQCO", "Q-MaskCut"],
              "q_maskcut": ("best-TVD key" if args.key_mode == "best" else "uniform random key") + ", H on decoys, CutQC 2 fragments, RX pair at cut, one layout per fragment",
              "after_compile_masking": ("time-aware Pauli twirl" if args.twirl else "no twirl") + " + " +
                                       (f"no-pattern DD ({args.dd_pairs} pairs/gap)" if args.dd else "no DD")}
    from qexp_store import RunStore
    RunStore(args.run, config)                                       # raises if the folder has other settings
    log = logging.getLogger("qmaskcut_eval"); log.setLevel(logging.INFO); log.propagate = False
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for h in (logging.FileHandler(Path(args.run) / f"log_{args.noise}_{time.strftime('%Y%m%d_%H%M%S')}.log"),
              logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt); log.addHandler(h)
    log.info("run %s | backend %s | %d circuits: %s", args.run, args.noise, len(names), " ".join(names))
    with Pool(min(args.workers, len(names)), initializer=_init, initargs=(args,)) as pool:
        for rows in pool.imap_unordered(one, names):
            for r in rows:
                if r.get("status") == "ok":
                    log.info("%-14s %-16s fid %.4f | TVD_obf_min %.3f | TVD_deobf %.3f | TVD_ideal %.1e | normGED %.3f | depth %s",
                             r["circuit"], r["method"], r["fidelity"], r["tvd_obf_min"], r["tvd_deobf"], r["tvd_ideal"],
                             r["normged"], r["depth"])
                else:
                    log.info("%-14s %-16s %s", r.get("circuit"), r.get("method"), r.get("status"))
    log.info("done")


if __name__ == "__main__":
    main()
