#!/usr/bin/env python3
"""Keeping the masking through compilation, on real hardware: barriers vs post-transpile masking.

Per circuit, four versions, all compiled at optimisation level 3 for the real backend with
couplers of error >= 0.5 left out:
  original                  the circuit itself
  as built                  Q-MaskCut; level 3 removes the idle-time masking (it is the identity)
  barriers                  Q-MaskCut with a one-qubit barrier after every masking gate
  post-transpile masking    Q-MaskCut compiled, THEN scheduled (ALAPScheduleAnalysis) and its idle
                            gaps filled (PadDynamicalDecoupling) with pulse PAIRS that each rotate by
                            pi about a RANDOM axis phi drawn from the client's seed:
                            rz(phi) x rz(-phi), twice = identity. rz is virtual (zero duration), so
                            timing is that of plain X-X, but no fixed pattern is visible.
One job, Runtime DD OFF (only our masking differs), gate + measurement twirling on, 4096 shots.
Before submitting, every version is checked noiselessly: decoded with the key, it must give
the ideal answer.

    .venv/bin/python scripts/hw_masking_test.py --account pknu2 --backend ibm_kingston [--submit|--collect]
"""
from __future__ import annotations
import argparse, csv, json, math, pickle, random, sys, time
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
bench = __import__("main-benchmark")
from qmaskcut import config as cfg
cfg.update(**bench.SETTINGS)
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import XGate
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDynamicalDecoupling
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qmaskcut.masking import protect_general
from qmaskcut.metrics.ideal import ideal
from qmaskcut.metrics.normged import deterministic_normged
from qmaskcut.utils.circuits import load, two_q_only
from hw_prepare import log_esp
from hw_rerun_original import working_map
from masking_survival_test import with_barriers

CIRCS = {"bell_n4": "small/bell_n4/bell_n4.qasm", "qpe_n9": "small/qpe_n9/qpe_n9.qasm"}
SHOTS, BEST_OF, SEED = 4096, 4, 42
OPTS = {"dynamical_decoupling": {"enable": False},
        "twirling": {"enable_gates": True, "enable_measure": True,
                     "num_randomizations": "auto", "shots_per_randomization": "auto"}}


def compile_fair(qc, backend, cmap):
    best = None
    for k in range(BEST_OF):
        tc = transpile(qc, basis_gates=["cz", "rz", "sx", "x", "id"], coupling_map=cmap,
                       optimization_level=3, seed_transpiler=SEED + k)
        sc = log_esp(tc, backend.target)
        if best is None or sc > best[1]:
            best = (tc, sc)
    return best[0]


def drop_short_pairs(padded, target, min_gap_us, every_wire=False):
    """Threshold rule: a masking pair whose whole idle gap (delay, X, delay, X, delay) is shorter
    than min_gap_us is removed and the gap restored as one delay of the same total length."""
    if not min_gap_us:
        return padded
    dt = target.dt
    seq = {}
    for k, ins in enumerate(padded.data):
        for q in ins.qubits:
            seq.setdefault(q, []).append(k)
    drop, replace = set(), {}
    best_short = {}                  # per wire: its longest gap that was dropped (every_wire rule)
    for q, ks in seq.items():
        i = 0
        while i + 4 < len(ks):
            w = [padded.data[k] for k in ks[i:i + 5]]
            names = [x.operation.name for x in w]
            if names == ["delay", "x", "delay", "x", "delay"] and all(
                    getattr(w[j].operation, "label", None) == "mask" for j in (1, 3)) and all(
                    len(x.qubits) == 1 for x in w):
                xdur = round(target["x"][(padded.find_bit(q).index,)].duration / dt)
                gap = sum(int(w[j].operation.duration) for j in (0, 2, 4)) + 2 * xdur
                if gap * dt * 1e6 < min_gap_us:
                    drop.update(ks[i:i + 5]); replace[ks[i]] = (q, gap)
                    if gap > best_short.get(q, (0, None))[0]:
                        best_short[q] = (gap, ks[i:i + 5])
                else:
                    best_short[q] = (float("inf"), None)          # this wire keeps a pair
                i += 5
            else:
                i += 1
    if every_wire:                    # a wire with no kept pair keeps its longest one
        for q, (gap, ks) in best_short.items():
            if ks is not None:
                drop.difference_update(ks); replace.pop(ks[0], None)
    out = padded.copy_empty_like()
    for k, ins in enumerate(padded.data):
        if k in replace:
            q, gap = replace[k]; out.delay(gap, q, unit="dt")
        elif k not in drop:
            out.append(ins.operation, ins.qubits, ins.clbits)
    return out


def post_transpile_masking(tc, target, seed, min_gap_us=0.0):
    """Schedule, fill idle gaps with X pairs (only gaps >= min_gap_us kept), then turn every pair
    into two pi rotations about one random axis phi (rz(phi) x rz(-phi)): identity, no pattern."""
    pm = PassManager([ALAPScheduleAnalysis(target=target),
                      PadDynamicalDecoupling(target=target, dd_sequence=[XGate(label="mask"), XGate(label="mask")])])
    padded = drop_short_pairs(pm.run(tc), target, min_gap_us)
    rng = random.Random(seed)
    out = padded.copy_empty_like()
    count = {}
    for ins in padded.data:
        op = ins.operation
        if op.name == "x" and getattr(op, "label", None) == "mask":
            q = ins.qubits[0]
            k = count.get(q, 0); count[q] = k + 1
            if k % 2 == 0:
                phi = rng.uniform(0, 2 * math.pi); count[(q, "phi")] = phi
            phi = count[(q, "phi")]
            out.rz(phi, q); out.x(q); out.rz(-phi, q)
        else:
            out.append(op, ins.qubits, ins.clbits)
    return out, sum(v for k, v in count.items() if not isinstance(k, tuple))


def decode(counts_le, n, info):
    real = info["real"] if info else list(range(n)); flip = info["flip"] if info else "0" * n
    dec = Counter()
    for b, v in counts_le.items():
        dec["".join(str(int(b[real[i]]) ^ int(flip[i])) for i in range(n))] += v
    return dec


def hellinger(dec, ref):
    tot = sum(dec.values())
    return sum(math.sqrt(dec.get(k, 0) / tot * r) for k, r in ref.items()) ** 2


def gates_only(c):
    o = QuantumCircuit(*c.qregs)
    for i in c.data:
        if i.operation.name not in ("barrier", "delay", "measure"):
            o.append(i.operation, i.qubits)
    return o


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", required=True); ap.add_argument("--backend", default="ibm_kingston")
    ap.add_argument("--submit", action="store_true"); ap.add_argument("--collect", action="store_true")
    a = ap.parse_args()
    OUT = ROOT / "results/hw_masking_test" / a.backend
    OUT.mkdir(parents=True, exist_ok=True)
    svc = QiskitRuntimeService(name=a.account); backend = svc.backend(a.backend)
    if a.collect:
        P = pickle.load(open(OUT / "prepared.pkl", "rb")); rec = json.loads((OUT / "job.json").read_text())
        res = svc.job(rec["job"]).result(); rows = []
        for k, it in enumerate(P["items"]):
            c = P["circuits"][k]
            bs = [getattr(res[k].data, cr.name).get_bitstrings() for cr in c.cregs]
            le = Counter("".join(b[i][::-1] for b in bs) for i in range(len(bs[0])))
            f = hellinger(decode(le, it["n"], it["info"]), P["ideal"][it["circuit"]])
            raw = hellinger(decode(le, it["n"], dict(it["info"], flip="0" * it["n"])), P["ideal"][it["circuit"]]) \
                if it["info"] else f
            rows.append({k_: it[k_] for k_ in ("circuit", "version", "depth", "duration_us", "x_pulses",
                                               "cz", "normGED_vs_original", "GED2Q_vs_original")}
                        | {"fidelity_without_key_hardware": round(raw, 4)}
                        | {"fidelity_hardware": round(f, 4)})
            print(rows[-1], flush=True)
        try:
            print("QPU seconds:", svc.job(rec["job"]).usage())
        except Exception:
            pass
        with open(OUT / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        sys.exit()
    cmap, bad = working_map(backend)
    sim = AerSimulator()
    circuits, items, ideals = [], [], {}
    for name, path in CIRCS.items():
        base = load(name, path); n = base.num_qubits
        ref, _ = ideal(name, base); ideals[name] = ref
        prot, tags, info = protect_general(base, n_decoy=2, n_padlock=2, reps=2, seed=cfg.SEED)
        b = base.copy(); b.measure_all()
        p = prot.copy(); p.measure_all()
        pb = with_barriers(prot, tags); pb.measure_all()
        c_orig = compile_fair(b, backend, cmap)
        c_prot = compile_fair(p, backend, cmap)
        c_bar = compile_fair(pb, backend, cmap)
        c_post, npulse = post_transpile_masking(c_prot, backend.target, seed=cfg.SEED)
        g_orig = gates_only(c_orig)
        for ver, c, inf in (("original", c_orig, None), ("as built", c_prot, info),
                            ("barriers", c_bar, info), ("post-transpile masking", c_post, info)):
            g = gates_only(c)
            # noiseless check: decoded with the key it must give the ideal answer; WITHOUT the key
            # (raw data wires) it must not - that shows the one-time-pad twirl survived compilation
            cnt = sim.run(c, shots=2000, seed_simulator=1).result().get_counts()
            le = Counter({k.replace(" ", "")[::-1]: v for k, v in cnt.items()})
            fid0 = hellinger(decode(le, n, inf), ref)
            raw0 = hellinger(decode(le, n, dict(inf, flip="0" * n)), ref) if inf else fid0
            dur = c.estimate_duration(backend.target, unit="u")
            it = {"circuit": name, "version": ver, "n": n, "info": inf, "depth": g.depth(),
                  "duration_us": round(dur, 2) if dur else "", "x_pulses": g.count_ops().get("x", 0),
                  "cz": g.count_ops().get("cz", 0),
                  "normGED_vs_original": round(deterministic_normged(g_orig, g), 4) if inf else 0.0,
                  "GED2Q_vs_original": round(deterministic_normged(two_q_only(g_orig), two_q_only(g)), 4) if inf else 0.0,
                  "noiseless_check": round(fid0, 4), "noiseless_without_key": round(raw0, 4),
                  "twirl_gates_written": sum(t == "twirl" for t in tags) if inf else 0}
            items.append(it); circuits.append(c)
            print(f"{name:8s} {ver:23s} depth {it['depth']:4d}  X {it['x_pulses']:4d}  cz {it['cz']:3d}  "
                  f"normGED {it['normGED_vs_original']:.3f}  dur {dur:6.2f}us | with key {fid0:.3f}  WITHOUT key {raw0:.3f}", flush=True)
    bad_checks = [i for i in items if i["noiseless_check"] < 0.95]
    pickle.dump({"circuits": circuits, "items": items, "ideal": ideals}, open(OUT / "prepared.pkl", "wb"))
    if bad_checks:
        print("NOISELESS CHECK FAILED, not submitting:", [(i["circuit"], i["version"]) for i in bad_checks]); sys.exit(1)
    if not a.submit:
        print("all noiseless checks OK; add --submit"); sys.exit()
    job = SamplerV2(mode=backend, options=OPTS).run(circuits, shots=SHOTS)
    (OUT / "job.json").write_text(json.dumps({"job": job.job_id(), "account": a.account, "backend": a.backend,
                                              "submitted": time.strftime("%Y-%m-%d %H:%M:%S"), "shots": SHOTS,
                                              "options": OPTS}, indent=1))
    print("submitted job", job.job_id())
