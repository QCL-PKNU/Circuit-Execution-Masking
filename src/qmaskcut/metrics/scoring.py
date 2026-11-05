"""Fidelity, TVD and counts conversion, all on little-endian keys (char q = qubit q)."""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

# qubits to score on for the circuit being run (cfg.SCORE_QUBITS), None = whole output
ON: list[int] | None = None
_REF: tuple = (None, None, None)


def marginal(p: dict[str, float], wires: list[int]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in p.items():
        key = "".join(k[q] for q in wires)
        out[key] = out.get(key, 0.0) + v
    return out


def _ref_on(ref):
    global _REF
    if _REF[0] is not ref or _REF[1] != ON:
        _REF = (ref, list(ON), marginal(ref, ON))
    return _REF[2]


def score(probs: dict[str, float], ref: dict[str, float]) -> dict:
    """Hellinger fidelity, TVD and F_top of a little-endian distribution vs the ideal."""
    if ON is not None:
        return _score(marginal(probs, ON), _ref_on(ref))
    return _score(probs, ref)


def _score(probs: dict[str, float], ref: dict[str, float]) -> dict:
    top = max(ref, key=ref.get)
    hf = sum(np.sqrt(max(probs.get(k, 0.0), 0.0) * r) for k, r in ref.items()) ** 2
    on = sum(abs(probs.get(k, 0.0) - r) for k, r in ref.items())
    off = sum(abs(v) for k, v in probs.items() if k not in ref)
    return {"fidelity": float(hf), "TVD": float(0.5 * (on + off)),
            "F_top": float(probs.get(top, 0.0)),
            "top": max(probs, key=probs.get) if probs else ""}


def score_dense(dv: np.ndarray, ref: dict[str, float]) -> dict:
    """score() on a dense little-endian vector, without building a 2**n dict."""
    if ON is not None:          # index bit q = qubit q, so axis n-1-q of the reshaped vector
        n = int(np.log2(len(dv)))
        t = dv.reshape([2] * n)
        m = t.sum(axis=tuple(a for a in range(n) if n - 1 - a not in ON))
        keep = sorted(ON, reverse=True)       # remaining axes, highest qubit first
        probs = {}
        for idx in np.ndindex(*m.shape):
            bit = dict(zip(keep, idx))
            probs["".join(str(bit[q]) for q in ON)] = float(m[idx])
        return _score(probs, _ref_on(ref))
    top = max(ref, key=ref.get)
    idx = {k: int(k[::-1], 2) for k in ref}
    hf = sum(np.sqrt(max(dv[i], 0.0) * ref[k]) for k, i in idx.items()) ** 2
    on = sum(abs(dv[i] - ref[k]) for k, i in idx.items())
    off = float(np.abs(dv).sum()) - sum(abs(dv[i]) for i in idx.values())
    n = int(np.log2(len(dv)))
    return {"fidelity": float(hf), "TVD": float(0.5 * (on + off)), "F_top": float(dv[idx[top]]),
            "top": format(int(np.argmax(dv)), f"0{n}b")[::-1]}


def counts_to_le(counts: dict, n: int, qc: QuantumCircuit | None = None) -> dict[str, float]:
    """qiskit counts -> little-endian probabilities over QUBITS 0..n-1.

    A counts key is ordered by classical bit, not by qubit. When `qc` has its own
    measurements each clbit is mapped back to the qubit it measured: ECQCO's output
    measures q[3] into c[0], and reading bits positionally scrambles the answer.
    """
    clbit_of = list(range(n))
    if qc is not None and qc.num_clbits:
        m = {qc.find_bit(i.qubits[0]).index: qc.find_bit(i.clbits[0]).index
             for i in qc.data if i.operation.name == "measure"}
        clbit_of = [m[q] for q in range(n)]
    tot = sum(counts.values())
    out: dict[str, float] = {}
    for k, v in counts.items():
        bits = k.replace(" ", "")[::-1]
        b = "".join(bits[clbit_of[q]] for q in range(n))
        out[b] = out.get(b, 0.0) + v / tot
    return out


def fmt(s: dict) -> str:
    return (f"fidelity {s['fidelity']:.4f} | TVD {s['TVD']:.4f} | F_top {s['F_top']:.4f}")
