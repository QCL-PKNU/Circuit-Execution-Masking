#!/usr/bin/env python3
"""Score wide circuits on 10-qubit marginals instead of the full 2**N vector.

Why windows
-----------
Three separate things break when a protected circuit passes about 25 wires:

* a dense 2**N vector is unaffordable (2**36 int64 is ~550 GB, which killed one run);
* scoring only the ideal's own support makes TVD incomparable between the two arms,
  because the baseline arm still has its whole measured distribution;
* filtering reconstructed outcomes by a fixed probability threshold silently empties the
  distribution once 2**n outcomes share the mass, which is what produced the bogus
  "TVD 1.000 -> 0.500" on ising_n26.

A window fixes all three. CutQC2's dynamic definition already computes marginals natively
via a qubit spec of "A" (active) and "M" (merged), so a 10-qubit window is 1024 numbers
however wide the circuit is, and BOTH arms are marginalised onto the same wires, which
makes fidelity and TVD directly comparable again.
"""
from __future__ import annotations

import numpy as np


def flat_order(cc) -> list[int]:
    """Original wire index at each position of the qubit spec."""
    order = cc.reconstruction_qubit_order
    out: list[int] = []
    for sub in cc.greedy_subcircuit_order:
        out.extend(order[sub])
    return out


def window_spec(cc, wires: list[int]) -> str:
    """'A' for every chosen wire, 'M' for the rest."""
    w = set(wires)
    return "".join("A" if q in w else "M" for q in flat_order(cc))


def window_probs(cc, wires: list[int]) -> dict[str, float]:
    """Marginal of the reconstruction over `wires`, keyed little-endian on those wires."""
    spec = window_spec(cc, wires)
    vec = cc.compute_probabilities(spec)
    active = [q for q in flat_order(cc) if q in set(wires)]
    k = len(active)
    out: dict[str, float] = {}
    for idx, v in enumerate(vec):
        if v <= 0:
            continue
        bits = format(idx, f"0{k}b")            # MSB first over the active positions
        key = {q: b for q, b in zip(active, bits)}
        out["".join(key[q] for q in sorted(active))] = (
            out.get("".join(key[q] for q in sorted(active)), 0.0) + float(v))
    return out


def marginalise(dist: dict[str, float], wires: list[int]) -> dict[str, float]:
    """Marginalise a little-endian full distribution onto `wires` (ascending)."""
    sel = sorted(wires)
    out: dict[str, float] = {}
    for k, v in dist.items():
        key = "".join(k[q] for q in sel)
        out[key] = out.get(key, 0.0) + v
    return out


def windows_for(n_data: int, size: int = 10, count: int = 3) -> list[list[int]]:
    """Evenly spread, non-overlapping windows over the data wires."""
    size = min(size, n_data)
    if n_data <= size:
        return [list(range(n_data))]
    starts = np.linspace(0, n_data - size, count).astype(int)
    return [list(range(s, s + size)) for s in sorted(set(starts.tolist()))]


def window_decode(cc, info, data_indices: list[int], n_data: int) -> dict[str, float]:
    """Reconstruct a window and apply the classical key to it.

    Does the twirl survive being applied to a marginal rather than the whole vector?
    Yes, and for a structural reason: the key is a per-wire XOR, so flipping bit q is a
    relabelling of outcomes that touches wire q and nothing else. Marginalising sums over
    the wires OUTSIDE the window, and flipping a bit before summing over it cannot change
    that sum. So marginalise-then-decrypt and decrypt-then-marginalise agree exactly.
    The decoys never enter: the window is built from data wires only, so they are among
    the wires summed out.
    """
    real = info["real"]
    flip = [int(c) for c in info["flip"]]
    wires = [real[q] for q in data_indices]
    raw = window_probs(cc, wires)
    order = sorted(range(len(wires)), key=lambda i: wires[i])   # window_probs sorts wires
    out: dict[str, float] = {}
    for key, v in raw.items():
        bits = [0] * len(wires)
        for pos, i in enumerate(order):
            bits[i] = int(key[pos]) ^ flip[data_indices[i]]
        k = "".join(map(str, bits))
        out[k] = out.get(k, 0.0) + v
    return out
