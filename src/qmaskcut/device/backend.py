"""Execution on a fake IBM backend, with AerSimulator for anything wider than the chip."""
from __future__ import annotations

import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from .. import config as cfg
from ..utils.circuits import unmeasured
from ..utils.logs import log
from ..metrics.scoring import counts_to_le, fmt, score

class Device:
    """A target device: a qiskit_ibm_runtime fake backend, or a real IBM backend by name.

    A real backend ("ibm_marrakesh", "ibm_fez", ...) is loaded through the saved account
    ("pknu") for its coupling map, basis and CALIBRATION noise only. Nothing is submitted:
    compiling and simulating against it costs no QPU time.
    """

    def __init__(self, name: str):
        if name.startswith("ibm_"):
            from qiskit_ibm_runtime import QiskitRuntimeService
            self.fake = QiskitRuntimeService(name="pknu").backend(name)
        else:
            from qiskit_ibm_runtime import fake_provider
            if not hasattr(fake_provider, name) and hasattr(fake_provider, name + "V2"):
                name += "V2"              # runtime 0.48 ships only the V2 classes
            self.fake = getattr(fake_provider, name)()
        self.width = self.fake.num_qubits
        self.noisy = AerSimulator.from_backend(self.fake)
        # wider than MPS_ABOVE: statevector with shot branching (shots share one noisy
        # trajectory until a noise event splits them); MPS with noise stalls at 22+ qubits
        self.noisy_wide = AerSimulator.from_backend(self.fake, method="statevector",
                                                    shot_branching_enable=True,
                                                    shot_branching_sampling_enable=True)
        noise = NoiseModel.from_backend(self.fake)
        self.wide = AerSimulator(noise_model=noise, method="matrix_product_state")
        self.basis = list(noise.basis_gates)
        self.ideal = AerSimulator(method="matrix_product_state")

    def compile(self, qc: QuantumCircuit) -> QuantumCircuit:
        c = qc.copy()
        if c.num_clbits == 0:
            c.measure_all()
        if c.num_qubits > self.width:      # too wide for the chip: basis only, no routing
            return transpile(c, basis_gates=self.basis, optimization_level=cfg.OPT_LEVEL,
                             seed_transpiler=cfg.SEED)
        return transpile(c, self.fake, optimization_level=cfg.OPT_LEVEL, seed_transpiler=cfg.SEED)

    def run(self, circuits, shots=None):
        """Noisy counts for each circuit, and the executed (transpiled) circuits."""
        shots = shots or cfg.SHOTS
        compiled = [self.compile(c) for c in circuits]
        out = []
        for c, t in zip(circuits, compiled):
            if c.num_qubits > self.width:
                log.info(f"note         : {c.num_qubits} wires > {self.width}-qubit "
                         f"{cfg.BACKEND}; this one runs on AerSimulator with the {cfg.BACKEND} "
                         f"noise model, no coupling map")
                sim = self.wide
            else:
                sim = self.noisy_wide if c.num_qubits > cfg.MPS_ABOVE else self.noisy
            out.append(sim.run(t, shots=shots, seed_simulator=cfg.SEED).result().get_counts())
        return out, compiled

    def where(self, n: int) -> str:
        return cfg.BACKEND if n <= self.width else f"Aer + {cfg.BACKEND} noise"


def semantics(qc: QuantumCircuit, n: int, ref: dict, dev: Device) -> float:
    """Noiseless fidelity to the ideal: 1.0 means the circuit still computes the same thing."""
    q = unmeasured(qc)
    if q.num_qubits <= 24:
        p = Statevector(q).probabilities()
        probs: dict[str, float] = {}
        for i in np.flatnonzero(p > 1e-14):
            k = format(int(i), f"0{q.num_qubits}b")[::-1][:n]
            probs[k] = probs.get(k, 0.0) + float(p[i])
        how = "statevector"
    else:
        c = qc.copy()
        if c.num_clbits == 0:
            c.measure_all()
        t = transpile(c, dev.ideal, optimization_level=0)
        cnt = dev.ideal.run(t, shots=20000, seed_simulator=cfg.SEED).result().get_counts()
        probs = counts_to_le(cnt, n, qc if qc.num_clbits else None)
        how = "20000 noiseless MPS shots"
    f = score(probs, ref)["fidelity"]
    ok = f > 0.999 if how == "statevector" else f > 0.95
    log.info(f"semantics    : noiseless fidelity to the ideal = {f:.4f} ({how}) -> "
             f"{'OK' if ok else 'BROKEN: the scheme changed the output'}")
    return f


def run_whole(dev: Device, qc: QuantumCircuit, n: int, ref: dict) -> dict:
    """One circuit, uncut, on the device, scored on its first n qubits."""
    t = time.time()
    (counts,), (tqc,) = dev.run([qc])
    s = score(counts_to_le(counts, n, qc if qc.num_clbits else None), ref)
    s.update(depth_exec=tqc.depth(), twoq_exec=tqc.num_nonlocal_gates(),
             backend=dev.where(qc.num_qubits))
    log.info(f"{'run':12s} : {s['backend']}, {cfg.SHOTS} shots, executed depth {tqc.depth()}, "
             f"{tqc.num_nonlocal_gates()} two-qubit after routing, {time.time() - t:.1f}s")
    log.info(f"result       : {fmt(s)}")
    return s
