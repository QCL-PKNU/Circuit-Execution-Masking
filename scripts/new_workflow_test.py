#!/usr/bin/env python3
"""New Q-MaskCut workflow on a fake backend, no QPU: masking moved AFTER compilation.

  2 encrypt (key) -> 3 camouflage (2 decoys, 2 CZ padlocks; NO idle masking here) -> 4 CutQC cut
  -> 5 instances -> 6 compile each (level 3, best of 4, broken couplers out, one layout per
  fragment) -> 7 ALAP schedule + our masking in idle gaps >= 1 us (rz(phi) x rz(-phi), random phi
  from the key seed) -> 8 run -> 9 CutQC rebuild -> 10 decode (drop decoys, XOR key)

Per circuit: depth (original, protected uncut, largest fragment; duration), normGED of the
compiled original vs (A) the whole protected circuit after step 7 and (B) each fragment after
step 7, and fidelity on the backend's noise model (Aer; idle delays get T1/T2 relaxation) for the
original, the cut pipeline without step 7 and with step 7, and without the key.
Also draws the compiled + masked circuits of the first circuit as gate DAGs.

    .venv/bin/python scripts/new_workflow_test.py [--backend FakeMontrealV2]
"""
from __future__ import annotations
import argparse, csv, math, random, statistics as st, subprocess, sys, tempfile
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
bench = __import__("main-benchmark")
from qmaskcut import config as cfg
cfg.update(**bench.SETTINGS)
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import XGate, RZGate
from qiskit.transpiler import PassManager, CouplingMap
from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDynamicalDecoupling
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import fake_provider
from qmaskcut.cutting import cutqc as C
from qmaskcut.masking import protect_general
from qmaskcut.metrics.ideal import ideal
from qmaskcut.metrics.normged import deterministic_normged
from qmaskcut.metrics.scoring import score_dense, score, counts_to_le
from qmaskcut.utils.circuits import load, two_q_only
from hw_prepare import log_esp
from hw_masking_test import drop_short_pairs

CIRCS = {"bell_n4": "small/bell_n4/bell_n4.qasm", "qpe_n9": "small/qpe_n9/qpe_n9.qasm",
         "adder_n10": "small/adder_n10/adder_n10.qasm", "seca_n11": "medium/seca_n11/seca_n11.qasm"}
SHOTS, BEST_OF, SEED, MIN_GAP_US = 4096, 4, 42, 1.0


def two_q(target):
    return next(g for g in ("cz", "ecr", "cx") if g in target.operation_names)


def working_map(target):
    g2 = two_q(target)
    edges = [qs for qs, p in target[g2].items() if p is None or p.error is None or p.error < 0.5]
    return CouplingMap(edges)


def tp(qc, target, cmap, seed, layout=None):
    return transpile(qc, basis_gates=[two_q(target), "rz", "sx", "x", "id"], coupling_map=cmap,
                     optimization_level=3, seed_transpiler=seed, initial_layout=layout)


def best(qc, target, cmap):
    b = None
    for k in range(BEST_OF):
        t = tp(qc, target, cmap, SEED + k); s = log_esp(t, target)
        if b is None or s > b[1]:
            b = (t, s)
    return b[0]


def mask(tc, target, seed, min_gap_us=None, pulses=2):
    """Step 7: schedule, X pairs in idle gaps, keep only gaps >= MIN_GAP_US, randomise the axis.
    Inserted gates are labelled 'mask' so the DAG can colour them."""
    gap = MIN_GAP_US if min_gap_us is None else min_gap_us
    pm = PassManager([ALAPScheduleAnalysis(target=target),
                      PadDynamicalDecoupling(target=target, dd_sequence=[XGate(label="mask") for _ in range(pulses)])])
    padded = pm.run(tc)
    if pulses == 2 and gap > 0:          # threshold rule (pairs only), every wire keeps one pair
        padded = drop_short_pairs(padded, target, gap, every_wire=True)
    rng = random.Random(seed); out = padded.copy_empty_like(); k = {}
    for ins in padded.data:
        op = ins.operation
        if op.name == "x" and getattr(op, "label", None) == "mask":
            q = ins.qubits[0]; i = k.get(q, 0); k[q] = i + 1
            if i % 2 == 0:
                k[(q, "p")] = rng.uniform(0, 2 * math.pi)
            phi = k[(q, "p")]
            out.append(RZGate(phi, label="mask"), [q]); out.append(XGate(label="mask"), [q])
            out.append(RZGate(-phi, label="mask"), [q])
        else:
            out.append(op, ins.qubits, ins.clbits)
    return out


def active_qubits(c):
    """Physical qubits the circuit really uses (any gate or measurement); idle chip qubits are
    never masked, they are not part of the submitted program."""
    return {c.find_bit(q).index for i in c.data if i.operation.name not in ("delay", "barrier") for q in i.qubits}


def mask_fill(tc, target, seed, max_pairs=4):
    """Step 7, fill version: schedule, then put into EVERY idle gap as many masking pulse pairs as
    fit (at least one pair needs 2 X durations; at most max_pairs pairs), evenly spaced; no
    extra time. Each pair becomes rz(phi) x rz(-phi) twice with a random phi from the seed."""
    from qiskit.transpiler.passes import PadDelay
    padded = PassManager([ALAPScheduleAnalysis(target=target), PadDelay(target=target)]).run(tc)
    dt = target.dt; act = active_qubits(padded)
    filled = padded.copy_empty_like()
    for ins in padded.data:
        op = ins.operation
        if op.name == "delay" and len(ins.qubits) == 1 and padded.find_bit(ins.qubits[0]).index in act:
            q = padded.find_bit(ins.qubits[0]).index
            d = int(op.duration) if op.unit == "dt" else int(round(op.duration / dt))
            xd = int(round(target["x"][(q,)].duration / dt))
            k = min(max_pairs, d // (2 * xd))
            if k >= 1:
                free = d - 2 * k * xd; slots = 2 * k + 1; base = free // slots
                gaps = [base] * slots; gaps[-1] += free - base * slots
                for j in range(2 * k):
                    if gaps[j]:
                        filled.delay(gaps[j], ins.qubits[0], unit="dt")
                    filled.append(XGate(label="mask"), ins.qubits)
                if gaps[-1]:
                    filled.delay(gaps[-1], ins.qubits[0], unit="dt")
                continue
        filled.append(op, ins.qubits, ins.clbits)
    rng = random.Random(seed); out = filled.copy_empty_like(); kk = {}
    for ins in filled.data:
        op = ins.operation
        if op.name == "x" and getattr(op, "label", None) == "mask":
            q = ins.qubits[0]; i = kk.get(q, 0); kk[q] = i + 1
            if i % 2 == 0:
                kk[(q, "p")] = rng.uniform(0, 2 * math.pi)
            phi = kk[(q, "p")]
            out.append(RZGate(phi, label="mask"), [q]); out.append(XGate(label="mask"), [q])
            out.append(RZGate(-phi, label="mask"), [q])
        else:
            out.append(op, ins.qubits, ins.clbits)
    return out


PAPER_UNITS = {   # the paper's idle-wire masking units (protect.py, reps=2), each A B A B = identity
    "XZ": [("x",), ("rz", math.pi)] * 2,
    "TX": [("rz", math.pi / 4), ("x",)] * 2,
    "SXZ": [("sx",), ("rz", math.pi)] * 2,
    "XY": [("x",), ("rz", math.pi), ("x",)] * 2,          # Y = i X Z, i.e. rz(pi) then x
}


def mask_pattern(tc, target, decoys=(), max_units=2):
    """Step 7, the paper's patterns after compilation: schedule, then put into every idle gap as
    many paper units (XZXZ on decoys; TXTX, SXZSXZ, XYXY cycling on the other wires) as fit, at
    most max_units, pulses evenly spaced; rz is virtual, so no extra time."""
    from qiskit.circuit.library import SXGate
    from qiskit.transpiler.passes import PadDelay
    padded = PassManager([ALAPScheduleAnalysis(target=target), PadDelay(target=target)]).run(tc)
    dt = target.dt; out = padded.copy_empty_like(); cyc = ["TX", "SXZ", "XY"]; act = active_qubits(padded)
    for ins in padded.data:
        op = ins.operation
        if op.name == "delay" and len(ins.qubits) == 1 and padded.find_bit(ins.qubits[0]).index in act:
            q = padded.find_bit(ins.qubits[0]).index
            unit = PAPER_UNITS["XZ" if q in decoys else cyc[q % 3]]
            d = int(op.duration) if op.unit == "dt" else int(round(op.duration / dt))
            xd = int(round(target["x"][(q,)].duration / dt)); per = sum(len(g) == 1 for g in unit)
            k = min(max_units, d // (per * xd))
            if k >= 1:
                seq = unit * k; npul = per * k; free = d - npul * xd; base = free // (npul + 1)
                gaps = [base] * (npul + 1); gaps[-1] += free - base * (npul + 1); j = 0
                for g in seq:
                    if len(g) == 1:
                        if gaps[j]:
                            out.delay(gaps[j], ins.qubits[0], unit="dt")
                        j += 1
                        out.append((XGate if g[0] == "x" else SXGate)(label="mask"), ins.qubits)
                    else:
                        out.append(RZGate(g[1], label="mask"), ins.qubits)
                if gaps[-1]:
                    out.delay(gaps[-1], ins.qubits[0], unit="dt")
                continue
        out.append(op, ins.qubits, ins.clbits)
    return out


def twirl_2q(tc, seed, paulis="IXYZ"):
    """Pauli twirl after compilation: a random Pauli pair P before every two-qubit gate G and
    G P G^dagger after it, so the product is unchanged. Native gates only: X -> x, Z -> rz(pi),
    Y -> rz(pi) x (equal up to global phase). Labelled "twirl". paulis="IZ" keeps it virtual
    (rz only: no time, no error)."""
    from qiskit.quantum_info import Clifford, Pauli
    rng = random.Random(seed); out = tc.copy_empty_like()

    def put(lbl, q):
        if lbl in "ZY":
            out.append(RZGate(math.pi, label="twirl"), [q])
        if lbl in "XY":
            out.append(XGate(label="twirl"), [q])
    for ins in tc.data:
        op = ins.operation
        if len(ins.qubits) == 2 and op.name not in ("barrier", "delay"):
            while True:
                p = "".join(rng.choice(paulis) for _ in range(2))
                if p != "II":
                    break
            after = Pauli(p).evolve(Clifford(op), frame="s").to_label().lstrip("-i")
            for k in range(2):                 # Pauli labels are little-endian: char -1-k is qubit k
                put(p[-1 - k], ins.qubits[k])
            out.append(op, ins.qubits, ins.clbits)
            for k in range(2):
                put(after[-1 - k], ins.qubits[k])
        else:
            out.append(op, ins.qubits, ins.clbits)
    return out


def twirl_smart(tc, target, seed):
    """Time-aware Pauli twirl after compilation. Around every two-qubit gate G a random Pauli
    pair P from {I,X,Z}^2 (no Y before the gate) and G P G^dagger after it. An X part is used on
    a qubit only if the idle time right before (or right after) G on that qubit holds the pulse,
    which is then taken out of that idle time; Z parts are virtual rz. So the schedule, and the
    run time, stay exactly as compiled. Labelled "twirl"."""
    from itertools import product
    from qiskit.quantum_info import Clifford, Pauli
    from qiskit.transpiler.passes import PadDelay
    padded = PassManager([ALAPScheduleAnalysis(target=target), PadDelay(target=target)]).run(tc)
    rng = random.Random(seed); dt = target.dt; data = list(padded.data)
    prev, nxt, last = {}, {}, {}
    for i, ins in enumerate(data):
        for q in ins.qubits:
            if q in last:
                prev[(i, q)] = last[q]; nxt[(last[q], q)] = i
            last[q] = i
    left = {i: (int(ins.operation.duration) if ins.operation.unit == "dt" else int(round(ins.operation.duration / dt)))
            for i, ins in enumerate(data) if ins.operation.name == "delay"}
    zs = {}
    for i, ins in enumerate(data):
        op = ins.operation
        if len(ins.qubits) != 2 or op.name in ("barrier", "delay"):
            continue
        qs = ins.qubits
        xd = [int(round(target["x"][(padded.find_bit(q).index,)].duration / dt)) for q in qs]
        def near(j, q, step):                 # the idle time next to G on q, past virtual rz gates
            while j is not None and data[j].operation.name == "rz":
                j = step.get((j, q))
            return j if j is not None and data[j].operation.name == "delay" else None
        dp = [near(prev.get((i, q)), q, prev) for q in qs]; dn = [near(nxt.get((i, q)), q, nxt) for q in qs]
        ok = []
        for p in product("IXZ", repeat=2):
            if p == ("I", "I"):
                continue
            after = Pauli(p[1] + p[0]).evolve(Clifford(op), frame="s").to_label().lstrip("-i")
            a = [after[-1 - k] for k in range(2)]
            if all((p[k] != "X" or (dp[k] in left and left[dp[k]] >= xd[k])) and
                   (a[k] not in "XY" or (dn[k] in left and left[dn[k]] >= xd[k])) for k in range(2)):
                ok.append((p, a))
        okx = [c for c in ok if "X" in c[0]]   # prefer an X twirl wherever it fits; Z-only always fits
        p, a = rng.choice(okx or ok)
        for k in range(2):                     # the pulse time is taken out of that idle time
            if p[k] == "X":
                left[dp[k]] -= xd[k]
            if a[k] in "XY":
                left[dn[k]] -= xd[k]
        zs[i] = (p, a)
    out = padded.copy_empty_like(); shrunk = {j for j in left if left[j] != (int(data[j].operation.duration) if data[j].operation.unit == "dt" else int(round(data[j].operation.duration / dt)))}
    for i, ins in enumerate(data):
        if i in shrunk:
            if left[i] > 0:
                out.delay(left[i], ins.qubits[0], unit="dt")
        elif i in zs:
            p, a = zs[i]
            for k, q in enumerate(ins.qubits):
                if p[k] == "X":
                    out.append(XGate(label="twirl"), [q])
                elif p[k] == "Z":
                    out.append(RZGate(math.pi, label="twirl"), [q])
            out.append(ins.operation, ins.qubits, ins.clbits)
            for k, q in enumerate(ins.qubits):     # Y = i X Z: rz(pi) first, then x
                if a[k] in "ZY":
                    out.append(RZGate(math.pi, label="twirl"), [q])
                if a[k] in "XY":
                    out.append(XGate(label="twirl"), [q])
        else:
            out.append(ins.operation, ins.qubits, ins.clbits)
    return out


def sched(tc, target):
    """Schedule as the device runs it: idle time becomes explicit delays (so a noise model charges
    idle decoherence to every arm alike)."""
    from qiskit.transpiler.passes import PadDelay
    return PassManager([ALAPScheduleAnalysis(target=target), PadDelay(target=target)]).run(tc)


def strip_mask(c):
    """c without its masking pulses, after checking that the pulses of every idle gap multiply to
    the identity (up to a global phase), so the stripped circuit computes exactly what c does."""
    import numpy as np
    from qiskit.quantum_info import Operator
    run, out = {}, c.copy_empty_like()

    def close(q):
        m = run.pop(q, None)
        if m is not None:
            ph = m[0, 0] if abs(m[0, 0]) > 1e-9 else 1
            assert np.allclose(m / ph, np.eye(2), atol=1e-9), f"masking on qubit {q} is not the identity"
    for i in c.data:
        q = [c.find_bit(x).index for x in i.qubits]
        if getattr(i.operation, "label", None) == "mask":
            run[q[0]] = Operator(i.operation).data @ run.get(q[0], np.eye(2)); continue
        if i.operation.name != "delay":
            for x in q:
                close(x)
        out.append(i.operation, i.qubits, i.clbits)
    for x in list(run):
        close(x)
    return out


def gates_only(c):
    o = QuantumCircuit(*c.qregs)
    for i in c.data:
        if i.operation.name not in ("barrier", "delay", "measure"):
            o.append(i.operation, i.qubits)
    return o


def wires_masked(c):
    used = {c.find_bit(q).index for i in c.data if i.operation.name not in ("barrier", "delay", "measure") for q in i.qubits}
    m = {c.find_bit(i.qubits[0]).index for i in c.data if getattr(i.operation, "label", None) == "mask"}
    return len(m & used), len(used)


def n_mask(c):
    return sum(1 for i in c.data if i.operation.name == "x" and getattr(i.operation, "label", None) == "mask")


def run_counts(sim, c):
    return sim.run(c, shots=SHOTS, seed_simulator=7).result().get_counts()


def rebuild(lab, cq, labels, circs, sim, info, n, ref):
    cq0 = C._finish(lab, cq.cut_solution["subcircuits"], cq.cut_solution["complete_path_map"])
    vecs = [C.counts_to_probs(run_counts(sim, c), c.num_clbits) for c in circs]
    C.evaluate_from(cq0, dict(zip(labels, vecs)))
    f = score_dense(C.full_decoded(cq0, info, n)[0], ref)["fidelity"]
    nokey = score_dense(C.full_decoded(cq0, dict(info, flip="0" * n), n)[0], ref)["fidelity"]
    return f, nokey


# ---------------------------------------------------------------- DAG drawing (same style as Fig. 2)
COL = {"orig": "#9fb4c7", "2q": "#e8b21e", "mask": "#a98fdb", "twirl": "#f0845c"}


def dag(c, title, decoys=(), cuts=()):
    phys = sorted({c.find_bit(q).index for i in c.data if i.operation.name not in ("barrier", "delay")
                   for q in i.qubits})
    L = ['digraph G {', 'rankdir=LR; ranksep=0.25; nodesep=0.14; newrank=true;',
         'node [shape=ellipse,style=filled,fontname="Helvetica",fontsize=11,width=0.52,height=0.34,fixedsize=true,penwidth=0.7];',
         'edge [arrowsize=0.5,color="#6b6b6b",penwidth=0.8];',
         f'labelloc="t"; labeljust="l"; label="{title}"; fontname="Helvetica-Bold"; fontsize=15;']
    last = {}
    for w in phys:
        dc = w in decoys
        L.append(f'in{w} [label="p{w}{chr(92)}n{"decoy" if dc else ""}",shape=box,style="filled,rounded",'
                 f'fillcolor="{"#5cb85c" if dc else "#eceff1"}",width=0.85,height=0.45];')
        last[w] = f"in{w}"
    for k, ins in enumerate(c.data):
        nm = ins.operation.name
        if nm in ("barrier", "delay", "measure"):
            continue
        qs = [c.find_bit(q).index for q in ins.qubits]
        lab = getattr(ins.operation, "label", None)
        if lab == "mask" and nm == "rz":      # the frame of a random-axis pulse: drawn as part of it
            continue
        col = COL[lab] if lab in ("mask", "twirl") else COL["2q" if len(qs) == 2 else "orig"]
        txt = "X&#966;" if lab == "mask" and nm == "x" else ("Z" if lab == "twirl" and nm == "rz" else nm)
        L.append(f'g{k} [label="{txt}",fillcolor="{col}"];')
        for w in qs:
            L.append(f'{last[w]} -> g{k};'); last[w] = f"g{k}"
    for w in phys:
        L.append(f'out{w} [label="p{w}",shape={"octagon" if w in cuts else "box"},style="filled,rounded",'
                 f'fillcolor="#ffdede",width=0.85,height=0.45];'); L.append(f'{last[w]} -> out{w};')
    L.append("}")
    return "\n".join(L)


def render(dots, stem):
    tmp = Path(tempfile.mkdtemp()); pngs = []
    for k, d in enumerate(dots):
        (tmp / f"{k}.dot").write_text(d)
        subprocess.run(["dot", "-Tpng", "-Gdpi=150", str(tmp / f"{k}.dot"), "-o", str(tmp / f"{k}.png")], check=True)
        pngs.append(tmp / f"{k}.png")
    from PIL import Image
    ims = [Image.open(p) for p in pngs]
    W = max(i.width for i in ims); H = sum(i.height for i in ims) + 20 * len(ims)
    canvas = Image.new("RGB", (W, H), "white"); y = 0
    for i in ims:
        canvas.paste(i, (0, y)); y += i.height + 20
    out = ROOT / "results/new_workflow" / f"{stem}.png"; canvas.save(out); return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--backend", default="FakeMontrealV2")
    a = ap.parse_args()
    be = getattr(fake_provider, a.backend)(); target = be.target; cmap = working_map(target)
    sim = AerSimulator.from_backend(be)
    (ROOT / "results/new_workflow").mkdir(parents=True, exist_ok=True)
    rows = []
    for ci, (name, path) in enumerate(CIRCS.items()):
        base = load(name, path); n = base.num_qubits; ref = ideal(name, base)[0]
        prot, tags, info = protect_general(base, n_decoy=2, n_padlock=2, reps=2, seed=cfg.SEED)
        lab = QuantumCircuit(*prot.qregs)                      # step 3 without idle masking
        for ins, t in zip(prot.data, tags):
            if t != "dd_pad":
                lab.append(ins.operation, ins.qubits)
        # original
        b = base.copy(); b.measure_all(); c_orig = best(b, target, cmap); g_orig = gates_only(c_orig)
        f_orig = score(counts_to_le(run_counts(sim, c_orig), n), ref)["fidelity"]
        # whole protected circuit (uncut), steps 6-7, for normGED A
        p = lab.copy(); p.measure_all(); c_whole = best(p, target, cmap); m_whole = mask(c_whole, target, cfg.SEED)
        ngA0 = deterministic_normged(g_orig, gates_only(c_whole)); ngA = deterministic_normged(g_orig, gates_only(m_whole))
        # cut, instances, compile (one layout per fragment), mask
        cq, _ = C.find_balanced_cut(lab, 2, max_cuts=2, allow_extra_cut=False)
        labels, frags = C.instance_circuits(cq)
        frags = [f if f.num_clbits else f.measure_all(inplace=False) for f in frags]
        comp = [None] * len(frags)
        for s in sorted({x[0] for x in labels}):
            idx = [j for j, x in enumerate(labels) if x[0] == s]
            rep = best(frags[idx[0]], target, cmap)
            lay = rep.layout.initial_index_layout(filter_ancillas=True)[:frags[idx[0]].num_qubits]
            for j in idx:
                comp[j] = tp(frags[j], target, cmap, SEED, layout=lay)
        masked = [mask(c, target, cfg.SEED + j) for j, c in enumerate(comp)]
        ngB0 = st.mean(deterministic_normged(g_orig, gates_only(c)) for c in comp)
        ngB = st.mean(deterministic_normged(g_orig, gates_only(c)) for c in masked)
        f_cut, _ = rebuild(lab, cq, labels, comp, sim, info, n, ref)
        f_mask, f_nokey = rebuild(lab, cq, labels, masked, sim, info, n, ref)
        big = max(masked, key=lambda c: gates_only(c).depth())
        row = {"circuit": name, "n": n, "K": cq.cut_solution["num_cuts"], "widths": "+".join(map(str, C.widths(cq))),
               "instances": len(frags),
               "depth_original": g_orig.depth(), "depth_protected_uncut": gates_only(c_whole).depth(),
               "depth_largest_fragment": gates_only(big).depth(),
               "duration_original_us": round(c_orig.estimate_duration(target, unit="u"), 2),
               "duration_largest_fragment_us": round(big.estimate_duration(target, unit="u"), 2),
               "mask_pulses_whole": n_mask(m_whole),
               "wires_masked_whole": "%d/%d" % wires_masked(m_whole),
               "wires_masked_fragments": "%d/%d" % tuple(map(sum, zip(*[wires_masked(c) for c in masked]))),
               "noiseless_check": round(rebuild(lab, cq, labels, masked, AerSimulator(), info, n, ref)[0], 3), "mask_pulses_per_fragment_mean": round(st.mean(n_mask(c) for c in masked), 1),
               "normGED_A_before_mask": round(ngA0, 3), "normGED_A_after_mask": round(ngA, 3),
               "normGED_B_before_mask": round(ngB0, 3), "normGED_B_after_mask": round(ngB, 3),
               "fid_original": round(f_orig, 3), "fid_cut_no_mask": round(f_cut, 3),
               "fid_cut_with_mask": round(f_mask, 3), "fid_without_key": round(f_nokey, 3)}
        rows.append(row); print(row, flush=True)
        if ci == 1:              # DAG of what is submitted, for the first circuit
            dec = {c_whole.layout.initial_index_layout(filter_ancillas=True)[q] for q in info["decoy"]}
            dots = [dag(c_orig, f"(a) ORIGINAL {name}, compiled for {a.backend}: depth {g_orig.depth()}"),
                    dag(m_whole, f"(b) PROTECTED, compiled + our masking after compile (purple): depth "
                                 f"{gates_only(m_whole).depth()}, {n_mask(m_whole)} pulses", decoys=dec)]
            for s in sorted({x[0] for x in labels}):
                j = next(j for j, x in enumerate(labels) if x[0] == s)
                dots.append(dag(masked[j], f"({chr(99 + s)}) FRAGMENT F{s + 1} (one instance), compiled + masking: "
                                           f"depth {gates_only(masked[j]).depth()}, {n_mask(masked[j])} pulses"))
            print("DAG:", render(dots, f"dag_{name}_{a.backend}"))
    with open(ROOT / "results/new_workflow" / f"results_{a.backend}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
