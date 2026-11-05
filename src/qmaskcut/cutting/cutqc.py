#!/usr/bin/env python3
"""CutQC (weiT1993/CutQC, vendored at <repo>/CutQC) for cutting AND reconstruction.

What CutQC does, unchanged
--------------------------
find_cuts (Gurobi MIP), subcircuits_parser, _generate_metadata (compute graph, entries,
instances), CutQC.evaluate (mutate_measurement_basis, attribute_shots),
DynamicDefinition.merge_states_into_bins and GraphContractor (the 4**K basis sum).

What is swapped in
------------------
* evaluate_circ, when fragments come from hardware: `evaluate_from` hands CutQC the
  measured vector for each (subcircuit, init, meas) instead of simulating it.
* merge_prob_vector, measure_prob and nearest_probability_distribution: NumPy versions with the same
  semantics. The originals are pure-Python loops over 2**width states and do not finish
  on a 20-wire fragment or a 29-wire vector. cutqc_hw_rebuild.py --check-merge and
  --check-nearest compare each against the original.

Gurobi: GRB_LICENSE_FILE points at the WLS licence <repo>/gurobi.lic when that file exists; a node-locked
licence is refused inside a container. Without any licence, gurobipy's size-limited licence is used.
"""
from __future__ import annotations

import os, sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if (_REPO / "gurobi.lic").exists():       # else Gurobi's own search; with no licence at all, gurobipy's built-in
    os.environ.setdefault("GRB_LICENSE_FILE", str(_REPO / "gurobi.lic"))   # size-limited licence cuts small circuits
sys.path.insert(0, str(_REPO / "CutQC"))

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler.passes import RemoveBarriers

import cutqc.evaluator as ev
import cutqc.dynamic_definition as dd
from cutqc.main import CutQC, _generate_metadata
from cutqc.cutter import get_pairs, get_counter, subcircuits_parser, read_circ, circuit_stripping
from cutqc.post_process_helper import get_reconstruction_qubit_order
from cutqc.graph_contraction import GraphContractor

_ORIG_EVAL = ev.evaluate_circ
_ORIG_MERGE = dd.merge_prob_vector


# ---------------------------------------------------------------- circuit preparation
def strip_for_cutting(qc: QuantumCircuit, unroll: bool = True) -> QuantumCircuit:
    """Put a QASMBench circuit into the form CutQC's check_valid accepts.

    Drops classical bits, measurements, barriers and resets, decomposes >2-qubit gates to
    {u3, cx}, and flattens everything onto ONE quantum register: CutQC names a gate
    vertex by register name and index, so several registers would still work, but a single
    register keeps the vertex names identical to the wire indices used everywhere else.
    """
    out = qc.remove_final_measurements(inplace=False)
    out = RemoveBarriers()(out)
    if unroll and any(len(i.qubits) > 2 for i in out.data):  # noqa: PLR2004
        out = transpile(out, basis_gates=["u3", "cx"], optimization_level=0)
    bare = QuantumCircuit(out.num_qubits)
    index = {q: i for i, q in enumerate(out.qubits)}
    for inst in out.data:
        if inst.operation.name in ("measure", "barrier", "reset"):
            continue
        bare.append(inst.operation, [bare.qubits[index[q]] for q in inst.qubits])
    return bare


# ---------------------------------------------------------------- cutting
def _finish(circuit, subcircuits, cpm) -> CutQC:
    counter = get_counter(subcircuits=subcircuits, O_rho_pairs=get_pairs(cpm))
    cq = CutQC(circuit=circuit, cutter_constraints={}, verbose=False)
    cq.cut_solution = {"subcircuits": subcircuits, "complete_path_map": cpm,
                       "num_cuts": len(get_pairs(cpm)), "counter": counter}
    cq.compute_graph, cq.subcircuit_entries, cq.subcircuit_instances = \
        _generate_metadata(counter, subcircuits, cpm)
    cq.times = {"cutter": 0.0}
    return cq


def cut_search(circuit, max_width: int, max_cuts: int, pieces: int) -> CutQC:
    """CutQC's own MIP search. Raises RuntimeError when there is no viable cut."""
    n_vertices = read_circ(circuit_stripping(circuit))[0]
    cq = CutQC(circuit=circuit, cutter_constraints={
        "max_subcircuit_width": max_width, "max_cuts": max_cuts,
        "num_subcircuits": [pieces], "max_subcircuit_cuts": max_cuts,
        # no gate-count balance: width is what the security argument balances
        "subcircuit_size_imbalance": max(n_vertices, 2) ** 2}, verbose=False)
    cq.cut()
    return cq


def find_balanced_cut(prot, pieces: int, max_cuts: int = 2, allow_extra_cut: bool = True):
    """Most balanced partition that exists, as (CutQC, width cap used).

    Balance is a SECURITY constraint: the cap starts at ceil((N+K)/pieces) and is relaxed
    one wire at a time, so the first success is the most even split available. With
    allow_extra_cut=False the search never exceeds max_cuts (CutQC treats it as a bound).
    """
    N = prot.num_qubits
    last = None
    for cap in range(-(-(N + max_cuts) // pieces), N + 1):
        for k in ((max_cuts, max_cuts + 1) if allow_extra_cut else (max_cuts,)):
            try:
                cq = cut_search(prot, cap, k, pieces)
            except Exception as e:      # noqa: BLE001 - this cap simply has no answer
                last = e
                continue
            if not collisions(cq):
                return cq, cap
            last = RuntimeError(f"cut at cap {cap} has colliding outputs")
    raise RuntimeError(f"no partition at any width cap: {last}")


def greedy_partition(circ, max_width: int):
    """Split the two-qubit gates, in circuit order, into chunks touching <= max_width wires.

    For chain- and star-shaped circuits (GHZ, cat, W-state, BV, Ising) consecutive chunks
    share one wire, so this is the minimum number of cuts, found in milliseconds. Returns
    CutQC vertex-name lists, or None if a single gate already exceeds the width.
    """
    _, _, _, id_vertices = read_circ(circuit_stripping(circ))
    names = [id_vertices[i] for i in sorted(id_vertices)]
    chunks, cur, wires = [], [], set()
    for v in names:
        q = {int(t.split("]")[0].split("[")[1]) for t in v.split(" ")}
        if cur and len(wires | q) > max_width:
            chunks.append(cur); cur, wires = [], set()
        cur.append(v); wires |= q
    if cur:
        chunks.append(cur)
    return chunks if len(chunks) > 1 else None


def cut_to_width(circ, max_width: int, extra_cuts: int = 1, max_pieces: int = 16):
    """Fewest fragments, then fewest cuts, with every fragment at most `max_width` wires.

    This is the capability use of cutting: a circuit too wide to simulate exactly (or to
    fit a device) becomes fragments that are not. The direct chain partition is tried
    first; CutQC's MIP search is the fallback for circuits it does not fit. Every
    candidate is rejected unless its fragments really are within the width and no two
    wires share a fragment output. Returns (CutQC, pieces).
    """
    chunks = greedy_partition(circ, max_width)
    if chunks:
        try:
            cq = cut_from_vertices(circ, chunks)
            if max(widths(cq)) <= max_width and not collisions(cq):
                return cq, len(chunks)
        except Exception:               # noqa: BLE001 - fall back to the MIP search
            pass
    N = circ.num_qubits
    last = None
    for pieces in range(max(2, -(-N // max_width)), max_pieces + 1):
        if pieces * max_width - (pieces - 1) < N:      # cannot cover N wires at this count
            continue
        for k in range(pieces - 1, pieces + extra_cuts):
            try:
                cq = cut_search(circ, max_width, k, pieces)
            except Exception as e:      # noqa: BLE001 - no solution at this (pieces, k)
                last = e
                continue
            if not collisions(cq):
                return cq, pieces
            last = RuntimeError(f"cut with {pieces} pieces has colliding outputs")
    raise RuntimeError(f"no partition within width {max_width}: {last}")


def independent_pieces(qc) -> list[list[int]]:
    """The circuit's independent pieces (wires linked by two-qubit gates), largest first.

    CutQC cuts connected circuits only. A circuit with idle or never-interacting wires
    (QASMBench BV with zero bits in its secret, ECQCO's Grover-7 and Shor-7) is already
    split; its largest piece is what gets cut.
    """
    parent = list(range(qc.num_qubits))
    find = lambda q: q if parent[q] == q else find(parent[q])         # noqa: E731
    for inst in qc.data:
        if len(inst.qubits) == 2:
            a, b = (find(qc.find_bit(q).index) for q in inst.qubits)
            parent[a] = b
    groups: dict[int, list[int]] = {}
    for q in range(qc.num_qubits):
        groups.setdefault(find(q), []).append(q)
    return sorted(groups.values(), key=len, reverse=True)


def piece_circuit(qc, wires: list[int]):
    """The gates of qc acting on `wires` (one independent piece), as a len(wires)-qubit circuit."""
    pos = {w: i for i, w in enumerate(wires)}
    out = QuantumCircuit(len(wires))
    for inst in qc.data:
        qs = [qc.find_bit(q).index for q in inst.qubits]
        if all(q in pos for q in qs):
            out.append(inst.operation, [pos[q] for q in qs])
    return out


def cut_from_vertices(circuit, subcircuit_vertices) -> CutQC:
    """Cut at an explicit gate partition: lists of CutQC vertex names "q[i]a q[j]b"."""
    subs, cpm = subcircuits_parser(subcircuit_gates=subcircuit_vertices, circuit=circuit)
    return _finish(circuit, subs, cpm)


def two_qubit_vertices(circuit) -> list[str]:
    """Every 2-qubit gate as a CutQC vertex name, in topological order."""
    _, _, _, id_vertices = read_circ(circuit_stripping(circuit))
    return [id_vertices[i] for i in sorted(id_vertices)]


def vertices_from_gate_split(circuit, splits):
    """Split the ordered 2-qubit gates at the given indices (the BV-style wire cut)."""
    names = two_qubit_vertices(circuit)
    splits = [splits] if isinstance(splits, int) else splits
    bounds = [0, *sorted(splits), len(names)]
    out = [names[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
    if any(not b for b in out):
        raise ValueError(f"split leaves an empty fragment: {[len(b) for b in out]}")
    return out


def from_solution(circuit, subcircuits, cpm_index) -> CutQC:
    """Rebuild a CutQC object from a saved partition: cpm as {wire: [(sub, sub_wire)]}."""
    cpm = {circuit.qubits[q]: [{"subcircuit_idx": s, "subcircuit_qubit": subcircuits[s].qubits[w]}
                               for s, w in path]
           for q, path in cpm_index.items()}
    return _finish(circuit, subcircuits, cpm)


def cpm_index(cq) -> dict[int, list[tuple[int, int]]]:
    sol = cq.cut_solution
    return {cq.circuit.qubits.index(q): [(p["subcircuit_idx"],
                                          sol["subcircuits"][p["subcircuit_idx"]].qubits.index(
                                              p["subcircuit_qubit"])) for p in path]
            for q, path in sol["complete_path_map"].items()}


def collisions(cq_or_cpm):
    """Fragment qubits that more than one original wire ends on (an invalid cut).

    cutqc2 0.0.9 produced one on adder_n10 (its own verify() reports error 1.50).
    """
    cpm = cq_or_cpm if isinstance(cq_or_cpm, dict) else cpm_index(cq_or_cpm)
    ends: dict[tuple, list[int]] = {}
    for q, path in cpm.items():
        ends.setdefault(tuple(path[-1]), []).append(q)
    return {k: v for k, v in ends.items() if len(v) > 1}


def widths(cq) -> list[int]:
    return [s.num_qubits for s in cq.cut_solution["subcircuits"]]


# ---------------------------------------------------------------- fragment instances
def instance_order(cq):
    """The instances CutQC.evaluate asks for, in the order it asks."""
    return [(s, tuple(i), tuple(m)) for s in range(len(cq.cut_solution["subcircuits"]))
            for i, m in cq.subcircuit_instances[s] if "Z" not in m]


def instance_circuits(cq):
    """(labels, circuits) to run on a device; same order as instance_order."""
    labels = instance_order(cq)
    subs = cq.cut_solution["subcircuits"]
    return labels, [ev.modify_subcircuit_instance(subcircuit=subs[s], init=i, meas=m)
                    for s, i, m in labels]


def evaluate_from(cq, vectors: dict):
    """CutQC.evaluate, with each instance's probability vector taken from `vectors`."""
    order = instance_order(cq)
    if set(order) != set(vectors):
        raise RuntimeError(f"instance sets differ: CutQC {len(order)}, given {len(vectors)}, "
                           f"only-CutQC {len(set(order) - set(vectors))}")
    queue = iter([vectors[k] for k in order])
    ev.evaluate_circ = lambda circuit, backend, options=None: next(queue)
    try:
        cq.evaluate(num_shots_fn=None)
    finally:
        ev.evaluate_circ = _ORIG_EVAL


def counts_to_probs(counts: dict, n: int) -> np.ndarray:
    """qiskit bitstring counts -> probability vector, little-endian index."""
    p = np.zeros(2 ** n)
    for k, v in counts.items():
        p[int(k.replace(" ", ""), 2)] += v
    s = p.sum()
    return p / s if s else p


# ---------------------------------------------------------------- reconstruction
def fast_merge(unmerged_prob_vector, qubit_states):
    """merge_prob_vector, vectorised. qubit_states[k] is bit k of the index, MSB first.

    float64 throughout; the original accumulates in float32, so the two agree to ~1e-7.
    """
    v = np.asarray(unmerged_prob_vector, dtype=float)
    if v.ndim == 0 or not qubit_states:
        return v.reshape(-1)
    t = v.reshape([2] * len(qubit_states))
    t = t[tuple(slice(None) if s in ("active", "merged") else int(s) for s in qubit_states)]
    kept = [s for s in qubit_states if s in ("active", "merged")]
    axes = tuple(i for i, s in enumerate(kept) if s == "merged")
    return np.ascontiguousarray(t.sum(axis=axes) if axes else t).reshape(-1)


dd.merge_prob_vector = fast_merge

_ORIG_MEASURE = ev.measure_prob


def fast_measure_prob(unmeasured_prob, meas):
    """measure_prob, vectorised; the same semantics as CutQC's.

    Bit j of the state index pairs with meas[j]. A position measured in X, Y or Z contributes
    a sign (-1)**bit, I and X/Y/Z positions are summed out, "comp" positions are kept in
    their order. CutQC's original walks every one of the 2**n states in Python, hours per
    instance at 27 wires; this is axis operations on the same vector.
    """
    if meas.count("comp") == len(meas) or type(unmeasured_prob) is float:
        return unmeasured_prob
    m = len(meas)
    t = np.array(unmeasured_prob, dtype=float).reshape([2] * m)     # axis a <-> bit m-1-a
    for j, b in enumerate(meas):
        if b not in ("I", "comp"):
            idx = [slice(None)] * m
            idx[m - 1 - j] = 1
            t[tuple(idx)] *= -1
    axes = tuple(m - 1 - j for j, b in enumerate(meas) if b != "comp")
    return np.ascontiguousarray(t.sum(axis=axes)).reshape(-1)


ev.measure_prob = fast_measure_prob


def nearest(quasiprobability):
    """CutQC's nearest_probability_distribution (Smolin et al. 2012), vectorised.

    Walking ascending, the dropped entries are always a prefix: once one entry is kept,
    beta stops changing and every larger entry is kept too. A stable argsort breaks ties
    by index, which is what sorting (prob, index) tuples does.
    """
    q = np.asarray(quasiprobability, dtype=float)
    M = len(q)
    order = np.argsort(q, kind="stable")
    p = q[order]
    beta = np.concatenate(([0.0], np.cumsum(p)[:-1]))      # sum of everything before k
    keep = p + beta / (M - np.arange(M)) >= 0
    m = int(np.argmax(keep)) if keep.any() else M
    out = np.zeros(M)
    if m < M:
        out[order[m:]] = p[m:] + beta[m] / (M - m)
    return out


def contract(cq, active: set[int]):
    """Marginal over the `active` wires of the cut circuit: (vector, wires MSB first)."""
    sol = cq.cut_solution
    out_order = get_reconstruction_qubit_order(cq.circuit, sol["complete_path_map"],
                                               sol["subcircuits"])
    sched = {"subcircuit_state": {s: ["active" if w in active else "merged" for w in ws]
                                  for s, ws in out_order.items()},
             "upper_bin": None}
    D = dd.DynamicDefinition(compute_graph=cq.compute_graph, num_cuts=sol["num_cuts"],
                             subcircuit_entry_probs=cq.subcircuit_entry_probs,
                             mem_limit=len(active), recursion_depth=1)
    gc = GraphContractor(compute_graph=cq.compute_graph,
                         subcircuit_entry_probs=D.merge_states_into_bins(sched),
                         num_cuts=sol["num_cuts"])
    wires = [w for s in gc.smart_order for w in out_order[s] if w in active]
    return np.asarray(gc.reconstructed_prob, dtype=float), wires


def decode_dense(vec, wires, info, qubits):
    """Map a reconstruction over `wires` onto original `qubits`, undoing the XOR key.

    vec is indexed MSB-first over `wires`; the result is little-endian over `qubits`, so
    key = format(i, f"0{len(qubits)}b")[::-1], char j being original qubit qubits[j].
    """
    real, flip = info["real"], [int(c) for c in info["flip"]]
    L, pos = len(wires), {w: j for j, w in enumerate(wires)}
    i = np.arange(2 ** L, dtype=np.int64)
    d = np.zeros_like(i)
    for j, q in enumerate(qubits):
        d |= (((i >> (L - 1 - pos[real[q]])) & 1) ^ flip[q]) << j
    return np.bincount(d, weights=vec, minlength=2 ** len(qubits))


def full_decoded(cq, info, n):
    """As CutQC's full_verify: every wire, nearest projection, THEN marginalise.

    Projecting after marginalising gives a slightly different answer (bv_n4 0.4845 vs
    0.4824), so the order matters for matching earlier numbers. Returns (projected,
    raw, negative mass), both dense over the n original qubits.
    """
    raw, wires = contract(cq, set(range(cq.circuit.num_qubits)))
    neg = float(raw[raw < 0].sum())
    q = list(range(n))
    return _unit(decode_dense(nearest(raw), wires, info, q)), decode_dense(raw, wires, info, q), neg


def _unit(dv):
    """Rescale to total probability 1. The projection keeps the quasi-probability vector's
    own total, which shot noise moves slightly away from 1; scoring an unnormalised vector
    can give a TVD above 1."""
    t = float(dv.sum())
    return dv / t if t > 0 else dv


def point_decoded(cq, info, outcome: str) -> float:
    """Probability of ONE whole original outcome, at any width, by CutQC's own contraction.

    `outcome` is little-endian over the original qubits. Every data wire is fixed to its
    bit XOR the key, every decoy wire is summed out, so the 4**K basis sum runs over one
    bin per fragment: no 2**N vector is formed. The value is the raw estimate of
    Eq. (1); a single point cannot be nearest-projected, so it is not.
    """
    sol = cq.cut_solution
    out_order = get_reconstruction_qubit_order(cq.circuit, sol["complete_path_map"],
                                               sol["subcircuits"])
    fixed = {info["real"][q]: str(int(b) ^ int(info["flip"][q])) for q, b in enumerate(outcome)}
    sched = {"subcircuit_state": {s: [fixed.get(w, "merged") for w in ws]
                                  for s, ws in out_order.items()},
             "upper_bin": None}
    D = dd.DynamicDefinition(compute_graph=cq.compute_graph, num_cuts=sol["num_cuts"],
                             subcircuit_entry_probs=cq.subcircuit_entry_probs,
                             mem_limit=1, recursion_depth=1)
    gc = GraphContractor(compute_graph=cq.compute_graph,
                         subcircuit_entry_probs=D.merge_states_into_bins(sched),
                         num_cuts=sol["num_cuts"])
    return float(np.asarray(gc.reconstructed_prob, dtype=float).sum())


def window_decoded(cq, info, qubits):
    """Marginal on a window of original qubits, nearest-projected on the window."""
    raw, wires = contract(cq, {info["real"][q] for q in qubits})
    return _unit(decode_dense(nearest(raw), wires, info, qubits))


def score_dense(dv, ref):
    """(F at the ideal dominant outcome, TVD, top) on a dense little-endian vector."""
    top = max(ref, key=ref.get)
    idx = {k: int(k[::-1], 2) for k in ref}
    on = sum(abs(dv[i] - ref[k]) for k, i in idx.items())
    off = np.abs(dv).sum() - sum(abs(dv[i]) for i in idx.values())
    return float(dv[idx[top]]), float(0.5 * (on + off)), top


def to_dict(dv, keep=200000):
    """Largest nonzero entries as {key: p}, the hw2 qmaskcut_probs format."""
    n = int(np.log2(len(dv)))
    nz = np.flatnonzero(dv)
    if len(nz) > keep:
        nz = nz[np.argsort(-np.abs(dv[nz]))[:keep]]
    return {format(int(i), f"0{n}b")[::-1]: float(dv[i]) for i in nz}
