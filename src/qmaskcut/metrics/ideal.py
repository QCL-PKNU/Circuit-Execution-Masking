#!/usr/bin/env python3
"""The ideal output distribution, so fidelity is scored against the RIGHT outcome.

The first report scored fidelity at the outcome the device returned most often. That is
not fidelity, it is just the tallest bar: on a noisy 9-qubit QPE run the device's
favourite string need not be the correct one, and the two arms can even peak on different
strings, which makes the comparison meaningless. Fidelity here is therefore always
P(the ideal dominant outcome), with the ideal computed independently of any run.
"""
from __future__ import annotations

STATEVECTOR_MAX = 26          # 2**26 complex128 is about 1 GB


def analytic(name: str, n: int) -> dict[str, float] | None:
    """Closed forms for the structured families, which is what survives past 26 qubits."""
    s = name.lower()
    if s.startswith(("ghz", "cat")):
        return {"0" * n: 0.5, "1" * n: 0.5}
    if s.startswith("bv_ones"):
        # n-1 secret bits all 1, ancilla left in |-> so its bit is free
        return {"1" * (n - 1) + "0": 0.5, "1" * (n - 1) + "1": 0.5}
    if s.startswith("wstate"):
        return {"0" * i + "1" + "0" * (n - i - 1): 1.0 / n for i in range(n)}
    return None


def mps_marginal(circuit, wires, shots=200000):
    """Reference marginal from a matrix-product-state simulation.

    For circuits with no closed form and too many qubits for a statevector, MPS is exact
    and cheap as long as entanglement stays low, which covers the families we cut:
    BV, GHZ, cat, W-state, kNN and Ising. It is sampled, so the reference carries its own
    1/sqrt(shots) noise and is not a substitute for an analytic answer where one exists.
    """
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    qc = circuit.copy()
    qc.measure_all()
    sim = AerSimulator(method="matrix_product_state")
    counts = sim.run(transpile(qc, sim), shots=shots).result().get_counts()
    sel = sorted(wires)
    out = {}
    for k, v in counts.items():
        b = k.replace(" ", "")[::-1]          # little-endian, qubit q at position q
        key = "".join(b[q] for q in sel)
        out[key] = out.get(key, 0.0) + v / shots
    return out


def ideal(name: str, circuit) -> tuple[dict[str, float] | None, str]:
    """Return (distribution keyed little-endian like Qiskit, how it was obtained)."""
    n = circuit.num_qubits
    a = analytic(name, n)
    if a is not None:
        return a, "analytic"
    if n <= STATEVECTOR_MAX:
        from qiskit.quantum_info import Statevector
        p = Statevector(circuit).probabilities()
        return ({format(i, f"0{n}b")[::-1]: float(v)
                 for i, v in enumerate(p) if v > 1e-12}, "statevector")
    return None, "unavailable"


def score(observed: dict[str, float], ref: dict[str, float]):
    """(fidelity at the ideal dominant outcome, TVD against the whole ideal)."""
    top = max(ref, key=ref.get)
    f = observed.get(top, 0.0)
    tvd = 0.5 * sum(abs(observed.get(k, 0.0) - ref.get(k, 0.0))
                    for k in set(observed) | set(ref))
    return f, tvd, top
