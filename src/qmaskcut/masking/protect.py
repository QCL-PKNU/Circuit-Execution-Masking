#!/usr/bin/env python3
"""Q-MaskCut protection for ARBITRARY circuits, via Pauli-frame propagation.

Why this exists
---------------
`protect_v2` is correct only for Bernstein-Vazirani. Its padlock is a single
CZ(decoy, data) with the decoy held in |1>, which acts as Z on the data wire, and a
mid-circuit Z is not identity. BV survives it because BV ends with H on every data wire
and H Z H = X, so the Z arrives at the measurement as a plain bit flip that folds into
the XOR key. No other benchmark hands you that.

The general rule
----------------
The Z does not need to meet an H. It only needs to stay a PAULI on its way to the
measurement; then its X-support is the XOR mask and the correction is again classical.
Propagation is exact through every Clifford, and Z passes straight through every diagonal
gate (rz, t, p, cz) because they commute. It breaks only on a non-Clifford rotation that
mixes X and Z, i.e. rx/ry/u3 with a generic angle, and only when the frame has an X part
on that wire at that moment.

Since we choose WHERE to put the padlock, a position that breaks is simply rejected and
the padlock slides later. In the worst case it lands immediately before the measurement,
where Z is exactly a no-op for a computational-basis readout, so a valid placement always
exists. This keeps the v2 single-CZ design (not identity, so an optimiser cannot delete
it) while making it correct for any circuit.

Masking is unchanged from v2: idle wires get two-gate blocks that square to identity, so
`reps=2` is exactly the identity and `reps=1` is not.
"""
from __future__ import annotations

import random as _r

from qiskit import QuantumCircuit
from qiskit.circuit.library import (CZGate, SXGate, TGate, XGate, YGate, ZGate)

# ---------------------------------------------------------------- Pauli frame
# A frame is (x, z): two bit-lists over the wires. Gate rules are the standard
# symplectic action of conjugation, P -> G P G^dagger.
DIAGONAL = {"rz", "t", "tdg", "s", "sdg", "z", "p", "u1", "cz", "cp", "crz"}
PAULI = {"x", "y", "z", "id"}


def _apply(gate: str, qargs: list[int], x: list[int], z: list[int]) -> bool:
    """Push the frame through one gate. False means the frame stops being a Pauli."""
    if gate in ("barrier", "measure", "delay", "reset"):
        return gate != "reset"
    q = qargs[0]

    if gate == "h":
        x[q], z[q] = z[q], x[q]; return True
    if gate in ("s", "sdg"):
        z[q] ^= x[q]; return True
    if gate in ("sx", "sxdg"):
        x[q] ^= z[q]; return True
    if gate in PAULI:
        return True                                   # Pauli conjugation is a phase only
    if gate in ("cx", "cnot"):
        c, t = qargs
        x[t] ^= x[c]; z[c] ^= z[t]; return True
    if gate == "cz":
        c, t = qargs
        z[c] ^= x[t]; z[t] ^= x[c]; return True
    if gate == "swap":
        a, b = qargs
        x[a], x[b] = x[b], x[a]; z[a], z[b] = z[b], z[a]; return True
    if gate in DIAGONAL:
        # commutes with Z but not with X
        return all(x[w] == 0 for w in qargs)
    # rx, ry, u, u3, r, and anything else: safe only if the frame is trivial there
    return all(x[w] == 0 and z[w] == 0 for w in qargs)


def _propagate(ops, start: int, wire: int, n_wires: int):
    """Start a Z on `wire` at instruction `start` and push it to the end.

    Returns the X-support at the measurement, or None if the frame broke.
    """
    x = [0] * n_wires
    z = [0] * n_wires
    z[wire] = 1
    for gate, qargs in ops[start:]:
        if not _apply(gate, qargs, x, z):
            return None
    return x


def _gap_starts(ops, wire: int) -> list[int]:
    """Insertion points that can differ for a frame started on `wire`.

    A frame on one wire is untouched by gates on other wires (it is zero there), so every
    position between two consecutive gates on `wire` gives the same result. Trying only the
    first position of each gap therefore picks exactly the same earliest position as trying
    all len(ops) + 1 of them, with far fewer propagations.
    """
    return [0] + [i + 1 for i, (_, qa) in enumerate(ops) if wire in qa]


# ---------------------------------------------------------------- protection
def protect_general(qc: QuantumCircuit, n_decoy: int = 2, n_padlock: int = 2,
                    reps: int = 2, seed: int = 42, n_twirl: int = 2):
    """Return (protected_circuit, tags, info).

    info['real']  wire carrying each original qubit
    info['decoy'] decoy wires
    info['flip']  XOR mask over the ORIGINAL qubits, from the Pauli frames
    info['padlock_positions'] where each padlock landed, and why
    """
    rng = _r.Random(seed)
    n = qc.num_qubits
    N = n + n_decoy
    real = sorted(rng.sample(range(N), n))            # interleaved, never contiguous
    DEC = sorted(set(range(N)) - set(real))
    pos = {i: real[i] for i in range(n)}

    out = QuantumCircuit(N)
    tags: list[str] = []

    def app(op, qs, tag):
        out.append(op, [out.qubits[q] for q in qs])
        tags.append(tag)

    # ---- layer 1: decoys to |1> so that CZ(decoy, data) acts as Z(data)
    for d in DEC:
        app(XGate(), [d], "decoy_init")

    # ---- layer 2: the original circuit, with idle wires masked during each 2q gate
    for inst in qc.data:
        if inst.operation.name in ("measure", "barrier"):
            continue
        qs = [pos[qc.find_bit(q).index] for q in inst.qubits]
        app(inst.operation, qs, "original")
        if len(qs) == 2:  # noqa: PLR2004
            busy = set(qs)
            idle = [w for w in range(N) if w not in busy]
            for k, w in enumerate(idle):
                for _ in range(reps):
                    if w in DEC:
                        app(XGate(), [w], "dd_pad"); app(ZGate(), [w], "dd_pad")
                    elif k % 3 == 0:
                        app(TGate(), [w], "dd_pad"); app(XGate(), [w], "dd_pad")
                    elif k % 3 == 1:
                        app(SXGate(), [w], "dd_pad"); app(ZGate(), [w], "dd_pad")
                    else:
                        app(XGate(), [w], "dd_pad"); app(YGate(), [w], "dd_pad")

    # ---- layer 3: padlocks, placed at the earliest position whose frame survives
    ops = [(i.operation.name, [out.find_bit(q).index for q in i.qubits])
           for i in out.data]
    data_wires = [pos[i] for i in range(n)]
    flip = [0] * n
    placements = []
    npl = min(n_padlock, len(data_wires), len(DEC) * len(data_wires))
    inserts: dict[int, list[tuple[int, int]]] = {}

    for k in range(npl):
        d = DEC[k % len(DEC)]
        t = data_wires[k % len(data_wires)]
        # candidate positions: after each instruction, earliest first
        chosen = None
        for p in _gap_starts(ops, t):
            mask = _propagate(ops, p, t, N)
            if mask is not None:
                chosen = (p, mask); break
        if chosen is None:
            placements.append((d, t, None, "no valid position"))
            continue
        p, mask = chosen
        inserts.setdefault(p, []).append((d, t))
        for i in range(n):
            if mask[pos[i]]:
                flip[i] ^= 1
        placements.append((d, t, p, f"frame survives from instruction {p}"))

    # ---- layer 1b: an EXPLICIT QOTP twirl.
    # The padlock frame alone is not a one-time pad. A padlock that has to sit late
    # arrives at the measurement still as a Z, and Z has no X-support, so it contributes
    # nothing to the key: measured on the first campaign, 10 of 12 circuits came out with
    # an all-zero mask and therefore no output blinding whatsoever. An explicit X on a
    # data wire is a guaranteed bit flip, so n_twirl of them give a key that is uniform on
    # those bits no matter where the padlocks landed. Placement uses the same frame ladder
    # as the padlocks, so the twirl sits as early as correctness allows rather than being
    # an obvious final layer.
    # Two twirls can produce the SAME X-support and cancel in the XOR, which puts the key
    # back to zero: measured on ising, where the frame spreads through the CX chain and
    # both twirls land on the same parity. So a twirl is kept only if it actually changes
    # the key, and otherwise another wire is tried.
    candidates = list(data_wires)
    rng.shuffle(candidates)
    if n_twirl == "random":
        # a one-time pad: every data wire is twirled with probability 1/2, so the key is a
        # uniformly random set of wires (at least one), not a fixed number of them
        n_twirl = max(1, sum(rng.random() < 0.5 for _ in candidates))
    twirls = []
    for t in candidates:
        if len(twirls) >= n_twirl:
            break
        for p in _gap_starts(ops, t):
            x = [0] * N; z = [0] * N; x[t] = 1
            if not all(_apply(g, qa, x, z) for g, qa in ops[p:]):
                continue
            delta = [1 if x[pos[i]] else 0 for i in range(n)]
            if not any(delta):
                # invisible HERE (e.g. a later H turns the X into a Z, a no-op at the
                # measurement): try a later position on the same wire, which can still
                # survive as a bit flip. Giving up on the wire left every Ising circuit
                # with an all-zero key, i.e. no output blinding at all.
                continue
            new_flip = [f ^ d for f, d in zip(flip, delta)]
            if new_flip == flip:
                continue                   # it cancels the key here: try later on this wire
            inserts.setdefault(p, []).append(("TWIRL", t))
            flip[:] = new_flip
            twirls.append((t, p))
            break

    # rebuild with the padlocks and twirls inserted
    final = QuantumCircuit(N)
    ftags: list[str] = []

    def fapp(op, qs, tag):
        final.append(op, [final.qubits[q] for q in qs])
        ftags.append(tag)

    for p in range(len(out.data) + 1):
        for d, t in inserts.get(p, []):
            if d == "TWIRL":
                fapp(XGate(), [t], "twirl")
            else:
                fapp(CZGate(), [d, t], "padlock")
        if p < len(out.data):
            inst = out.data[p]
            fapp(inst.operation, [out.find_bit(q).index for q in inst.qubits], tags[p])

    return final, ftags, {
        "real": real, "decoy": DEC, "flip": "".join(map(str, flip)),
        "n_padlock": npl, "padlock_positions": placements,
        "twirls": twirls,
    }


def target_states(ref: dict, info, n_data: int, N: int):
    """The protected-register indices that decode to the ideal outcomes.

    Enumerating all 2**N outcomes is what killed the 34-qubit run: a 36-wire register is
    2**36 int64, about 550 GB. Only the ideal's own support is ever scored, and the decoy
    wires are free, so this is |ideal| * 2**n_decoy indices however wide the circuit is.
    """
    real = info["real"]
    flip = [int(c) for c in info["flip"]]
    free = [w for w in range(N) if w not in real[:n_data]]
    out = []
    for key in ref:
        base = 0
        for q in range(n_data):
            if int(key[q]) ^ flip[q]:
                base |= 1 << real[q]
        for m in range(2 ** len(free)):
            idx = base
            for j, w in enumerate(free):
                if (m >> j) & 1:
                    idx |= 1 << w
            out.append(idx)
    return sorted(set(out))


def decode_sparse(indices, probs, info, n_data: int, N: int):
    """Same as decode, but over an explicit index list rather than a dense vector."""
    real = info["real"]
    flip = [int(c) for c in info["flip"]]
    out: dict[str, float] = {}
    for idx, v in zip(indices, probs):
        if v <= 0:
            continue
        bits = [(int(idx) >> j) & 1 for j in range(N)]
        key = "".join(str(bits[real[q]] ^ flip[q]) for q in range(n_data))
        out[key] = out.get(key, 0.0) + float(v)
    return out


def decode(counts_or_probs, info, n_data: int, N: int):
    """Marginalise a protected-register distribution onto the original qubits."""
    real = info["real"]
    flip = [int(c) for c in info["flip"]]
    out: dict[str, float] = {}
    items = (counts_or_probs.items() if isinstance(counts_or_probs, dict)
             else enumerate(counts_or_probs))
    for idx, v in items:
        if v <= 0:
            continue
        i = int(idx, 2) if isinstance(idx, str) else int(idx)
        bits = [(i >> j) & 1 for j in range(N)]
        key = "".join(str(bits[real[q]] ^ flip[q]) for q in range(n_data))
        out[key] = out.get(key, 0.0) + float(v)
    return out
