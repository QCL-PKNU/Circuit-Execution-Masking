"""Readout-error correction, applied identically to both arms of a hardware run.

Each physical qubit's measurement is modelled as a symmetric bit flip with probability e
(the mean of IBM's prob_meas1_prep0 and prob_meas0_prep1). Measurement twirling, which the
mitigated runs enable, makes the real error symmetric, so the model fits. The correction
is the exact inverse of the tensored flip channel, so it returns quasi-probabilities.
"""
from __future__ import annotations

import numpy as np


def calibration(backend) -> dict[int, float]:
    """{physical qubit: symmetric readout flip probability} from the live properties."""
    pr = backend.properties()
    out = {}
    for q in range(backend.num_qubits):
        try:
            e = 0.5 * (pr.qubit_property(q, "prob_meas1_prep0")[0]
                       + pr.qubit_property(q, "prob_meas0_prep1")[0])
        except Exception:                    # noqa: BLE001 - fall back to readout_error
            e = pr.readout_error(q)
        out[q] = float(min(e, 0.45))
    return out


def clbit_qubits(tc) -> list[int]:
    """Physical qubit measured into each classical bit of a compiled circuit."""
    phys = [None] * tc.num_clbits
    for ins in tc.data:
        if ins.operation.name == "measure":
            phys[tc.find_bit(ins.clbits[0]).index] = tc.find_bit(ins.qubits[0]).index
    return phys


def _inv(e):
    return np.array([[1 - e, -e], [-e, 1 - e]]) / (1 - 2 * e)


def correct_dense(vec, errors: list[float]) -> np.ndarray:
    """Invert the readout flips on a probability vector whose index bit j is clbit j."""
    m = len(errors)
    t = np.asarray(vec, dtype=float).reshape([2] * m)      # axis 0 = most significant bit
    for j, e in enumerate(errors):
        t = np.moveaxis(np.tensordot(_inv(e), t, axes=([1], [m - 1 - j])), 0, m - 1 - j)
    return t.reshape(-1)


def correct_points(counts: dict, errors: list[float], outcomes: list[str]) -> dict[str, float]:
    """Corrected probability of chosen outcomes, at any width (no 2**m vector).

    `outcomes` are little-endian strings over clbits (char j = clbit j); counts keys are
    qiskit's big-endian bitstrings. P(x) = sum_y p(y) prod_j Ainv_j[x_j, y_j].
    """
    tot = sum(counts.values())
    inv = [_inv(e) for e in errors]
    obs = [(k.replace(" ", "")[::-1], v / tot) for k, v in counts.items()]
    out = {}
    for x in outcomes:
        s = 0.0
        for y, p in obs:
            w = p
            for j in range(len(x)):
                w *= inv[j][int(x[j]), int(y[j])]
            s += w
        out[x] = float(s)
    return out
