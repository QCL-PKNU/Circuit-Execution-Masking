"""The cut study and the Q-MaskCut arm: cut, check, run, reconstruct, decode."""
from __future__ import annotations

import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from .. import config as cfg
from ..cutting import cutqc as C
from ..utils.circuits import n2q
from ..device.backend import Device
from ..utils.logs import banner, block, describe, draw, log
from ..metrics.normged import structure
from ..masking.protect import protect_general
from ..metrics.scoring import fmt, score_dense

IDENTITY = lambda n: {"real": list(range(n)), "flip": "0" * n}     # noqa: E731


def cut_map(cq, N: int, info) -> str:
    """ASCII map of every wire's path through the fragments; X marks a wire cut."""
    cpm = C.cpm_index(cq)
    lines = []
    for q in range(N):
        role = f"data q{info['real'].index(q)}" if q in info["real"] else "decoy"
        lines.append(f"wire {q:2d} ({role:>8s}) : "
                     + " --X-- ".join(f"[F{s}:w{w}]" for s, w in cpm[q]))
    return "\n".join(lines)


def check_cut(cq, circ: QuantumCircuit, width: int) -> dict:
    subs = cq.cut_solution["subcircuits"]
    K, N, w = cq.cut_solution["num_cuts"], circ.num_qubits, C.widths(cq)
    t2f, t2 = sum(n2q(s) for s in subs), n2q(circ)
    of = sum(s.size() for s in subs)
    checks = {
        "widths sum to N + K": sum(w) == N + K,
        "every 2-qubit gate in exactly one fragment": t2f == t2,
        "every gate kept": of == circ.size(),
        "no two wires end on one fragment qubit": not C.collisions(cq),
        f"fragments fit the {width}-qubit {cfg.BACKEND}": max(w) <= width,
    }
    for k, ok in checks.items():
        log.info(f"cut check    : {k:45s} {'OK' if ok else 'FAILED'}")
    log.info(f"               (sum widths {sum(w)} = {N} + {K}; 2-qubit gates {t2f}/{t2}; "
             f"gates {of}/{circ.size()})")
    return checks


def exact_cut_error(cq, circ, info, n, ref) -> float:
    """Noiseless CutQC reconstruction against the truth.

    Up to 24 wires against the circuit's own statevector; wider, the decoded
    reconstruction against the ideal distribution of the original circuit.
    """
    ex = C.from_solution(circ, cq.cut_solution["subcircuits"], C.cpm_index(cq))
    ex.evaluate(num_shots_fn=None)
    N = circ.num_qubits
    if N <= 24:
        vec, wires = C.contract(ex, set(range(N)))
        i = np.arange(2 ** N, dtype=np.int64)
        perm = np.zeros_like(i)
        for j, w in enumerate(wires):
            perm |= ((i >> (N - 1 - j)) & 1) << w
        err = float(np.abs(vec - Statevector(circ).probabilities()[perm]).max())
        what = "reconstruction vs statevector"
    else:
        dv = C.full_decoded(ex, info, n)[1]
        r = np.zeros(2 ** n)
        for k, v in ref.items():
            r[int(k[::-1], 2)] = v
        err = float(np.abs(dv - r).max())
        what = "decoded reconstruction vs ideal"
    log.info(f"exact cut    : max |{what}| = {err:.2e}  -> {'OK' if err < 1e-6 else 'FAILED'}")
    return err


def cut_and_run(dev: Device, circ: QuantumCircuit, pieces: int, info, n: int, ref: dict,
                tag: str) -> tuple[dict, object]:
    """Cut `circ` with CutQC, check it, run every fragment instance, reconstruct, decode."""
    log.info(f"[cut] CutQC into {pieces} fragments, at most {cfg.MAX_CUTS} cuts (Gurobi MIP)")
    t = time.time()
    cq, cap = C.find_balanced_cut(circ, pieces, max_cuts=cfg.MAX_CUTS, allow_extra_cut=False)
    subs = cq.cut_solution["subcircuits"]
    K, N = cq.cut_solution["num_cuts"], circ.num_qubits
    labels, frags = C.instance_circuits(cq)
    log.info(f"partition    : widths {C.widths(cq)} (= {'+'.join(map(str, C.widths(cq)))}), "
             f"K = {K} cut(s), width cap {cap}, {time.time() - t:.1f}s")
    log.info("cut map      : each wire's path through the fragments (X = cut)")
    block(cut_map(cq, N, info))
    for i, s in enumerate(subs):
        log.info(f"fragment {i}   : {s.num_qubits} qubits, {s.size()} gates, depth {s.depth()}, "
                 f"{n2q(s)} two-qubit")
        draw(s, f"fragment {i}")
    log.info(f"instances    : {len(frags)} circuits to execute (init/meas variants, 4^K terms)")
    rec = {"widths": C.widths(cq), "pieces": "+".join(map(str, C.widths(cq))), "K": K,
           "instances": len(frags)}
    log.info("[check] the cut is a correct decomposition")
    rec["cut_checks"] = check_cut(cq, circ, dev.width)
    rec["cut_exact_err"] = exact_cut_error(cq, circ, info, n, ref)

    log.info(f"[run] every fragment instance on {cfg.BACKEND}")
    t = time.time()
    counts, compiled = dev.run(frags)
    rec["depth_exec"] = max(c.depth() for c in compiled)
    rec["twoq_exec"] = max(c.num_nonlocal_gates() for c in compiled)
    log.info(f"{'run':12s} : {len(frags)} circuits x {cfg.SHOTS} shots, {time.time() - t:.1f}s; "
             f"executed depth (largest fragment) {rec['depth_exec']}")
    log.info("[reconstruct] CutQC")
    t = time.time()
    C.evaluate_from(cq, dict(zip(labels, [C.counts_to_probs(c, f.num_qubits)
                                          for c, f in zip(counts, frags)])))
    dv, _, neg = C.full_decoded(cq, info, n)
    log.info(f"reconstruct  : {time.time() - t:.1f}s, negative quasi-probability mass "
             f"{neg:.4f} (projected to the nearest distribution)")
    rec.update(score_dense(dv, ref), backend=cfg.BACKEND, negative_mass=neg)
    log.info(f"result       : {fmt(rec)}  [{tag}]")
    return rec, cq


# ================================================================ arms
def run_cut_study(dev, base, n, pieces, ref, uncut) -> dict:
    banner(f"{base.name} | CUT STUDY: unprotected, uncut vs {cfg.MAX_CUTS} cuts", "-")
    log.info(f"uncut        : {fmt(uncut)}  (from the baseline run above)")
    try:
        cut, _ = cut_and_run(dev, base, pieces, IDENTITY(n), n, ref, "cut")
    except Exception as e:      # noqa: BLE001 - one circuit failing must not stop the rest
        log.info(f"FAILED       : {type(e).__name__}: {e}")
        return {"scheme": "cut study", "error": str(e)}
    gain = cut["fidelity"] - uncut["fidelity"]
    log.info(f"gain         : fidelity {uncut['fidelity']:.4f} -> {cut['fidelity']:.4f} "
             f"({gain:+.4f}); TVD {uncut['TVD']:.4f} -> {cut['TVD']:.4f}")
    return {"scheme": "cut study", "uncut": uncut, "cut": cut, "gain": gain}


def run_qmaskcut(dev, base, n, pieces, ref) -> dict | None:
    banner(f"{base.name} | Q-MaskCut", "-")
    log.info("[protect] decoy wires, idle-wire padding, CZ padlocks, QOTP twirl")
    t0 = time.time()
    prot, tags, info = protect_general(base, n_decoy=cfg.QMC_DECOYS, n_padlock=cfg.QMC_PADLOCKS,
                                       reps=2, seed=cfg.SEED)
    N = prot.num_qubits
    log.info(f"wires        : {n} -> {N}  (real on {info['real']}, decoys on {info['decoy']})")
    log.info(f"gate roles   : { {t: tags.count(t) for t in dict.fromkeys(tags)} }")
    for d, t, p, _ in info["padlock_positions"]:
        log.info(f"padlock      : CZ(decoy {d}, data {t}) at instruction {p}")
    log.info(f"twirls       : X on (wire, instruction) {info['twirls']}")
    log.info(f"XOR key      : {info['flip']}  (held by the client only)")
    describe("protected", prot)
    draw(prot, "protected circuit (what the cloud receives, before cutting)")
    rec = {"scheme": "Q-MaskCut", "qubits": N, "gates": prot.size(), "depth": prot.depth(),
           "twoq": n2q(prot), "compile_s": time.time() - t0, "flip": info["flip"]}
    log.info("[structure] what the cloud's circuit graph reveals")
    rec.update(structure(base, prot))
    try:
        cut, cq = cut_and_run(dev, prot, pieces, info, n, ref, "client, with key")
    except Exception as e:      # noqa: BLE001
        log.info(f"FAILED       : {type(e).__name__}: {e}")
        return None
    rec.update(cut, semantics=1.0 if cut["cut_exact_err"] < 1e-6 else 0.0)
    cloud = score_dense(C.full_decoded(cq, dict(info, flip="0" * n), n)[0], ref)
    log.info(f"cloud view   : without the key, {fmt(cloud)}")
    rec["cloud"] = cloud
    t = time.perf_counter()
    _ = np.arange(min(2 ** n, 1 << 20)) ^ int(info["flip"][::-1], 2)
    log.info(f"client decode: one XOR with the key, {(time.perf_counter() - t) * 1e6:.1f} us")
    return rec


def run_circuit(dev: Device, name: str, path: str | None, pieces: int, selected: bool,
                out_dir) -> dict:
    """Every enabled arm on one circuit; returns its results record."""
    from ..baselines import OBFUSQATE, run_ecqco, run_obfusqate
    from ..utils.circuits import load
    from ..device.backend import run_whole, semantics
    from ..metrics.ideal import ideal

    base = load(name, path)
    base.name = name
    n = base.num_qubits
    ref, how = ideal(name, base)
    from ..metrics import scoring
    scoring.ON = cfg.SCORE_QUBITS.get(name)
    if scoring.ON is not None:
        log.info(f"scored on    : qubits {scoring.ON} (what the QASMBench file measures); "
                 f"ideal there {scoring._ref_on(ref)}")
    describe("original", base)
    draw(base, "original circuit")
    log.info(f"ideal        : {len(ref)} outcomes with p > 0 ({how}); most likely "
             f"'{max(ref, key=ref.get)}' p = {max(ref.values()):.4f}")
    rec = {"name": name, "n": n, "pieces_req": pieces, "selected": selected, "schemes": {}}

    log.info("unprotected baseline:")
    b = {"scheme": "unprotected", "qubits": n, "gates": base.size(), "depth": base.depth(),
         "twoq": n2q(base), "normGED": 0.0, "GED_2Q": 0.0,
         "semantics": semantics(base, n, ref, dev)}
    b.update(run_whole(dev, base, n, ref))
    rec["schemes"]["unprotected"] = b
    if cfg.RUN_CUT_STUDY:
        rec["cut study"] = run_cut_study(dev, base, n, pieces, ref, b)
    if cfg.RUN_QMASKCUT and (a := run_qmaskcut(dev, base, n, pieces, ref)):
        rec["schemes"]["Q-MaskCut"] = a
    if cfg.RUN_ECQCO and (a := run_ecqco(dev, base, n, ref, out_dir / "ecqco")):
        rec["schemes"]["ECQCO"] = a
    if cfg.RUN_OBFUSQATE:
        for tech in getattr(cfg, "OBFUSQATE_ARMS", list(OBFUSQATE)):
            if a := run_obfusqate(dev, base, n, ref, tech):
                rec["schemes"][tech] = a
    return rec
