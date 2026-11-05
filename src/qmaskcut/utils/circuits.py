"""Benchmark circuits and small circuit helpers."""
from __future__ import annotations

from qiskit import QuantumCircuit

from .. import config as cfg
from ..cutting import cutqc as C

def bv_ones(n: int) -> QuantumCircuit:
    """BV with secret '1'*(n-1); qubit n-1 is the ancilla."""
    qc = QuantumCircuit(n)
    qc.x(n - 1)
    qc.h(range(n))
    for i in range(n - 1):
        qc.cx(i, n - 1)
    qc.h(range(n - 1))
    return qc


def load(name: str, path: str | None) -> QuantumCircuit:
    qc = bv_ones(int(name.split("_n")[-1])) if path is None else \
        QuantumCircuit.from_qasm_file(str(cfg.QASMBENCH / path))
    return C.strip_for_cutting(qc)


def two_q_only(qc: QuantumCircuit) -> QuantumCircuit:
    out = QuantumCircuit(qc.num_qubits)
    for inst in qc.data:
        if len(inst.qubits) == 2:
            out.append(inst.operation, [qc.find_bit(q).index for q in inst.qubits])
    return out


def unmeasured(qc: QuantumCircuit) -> QuantumCircuit:
    return qc.remove_final_measurements(inplace=False)


def n2q(qc: QuantumCircuit) -> int:
    return sum(len(i.qubits) == 2 for i in qc.data)
