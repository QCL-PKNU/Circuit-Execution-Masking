"""ECQCO baseline: QCOO (QOTP) + QCSO (ADOA), run in its own ecqco-env."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, qasm2

from .. import config as cfg
from ..utils.circuits import n2q, unmeasured
from ..device.backend import run_whole, semantics
from ..utils.logs import banner, describe, draw, log
from ..metrics.normged import structure

def run_ecqco(dev, base, n, ref, work: Path) -> dict | None:
    banner(f"{base.name} | ECQCO (QCOO + QCSO/ADOA; ADOA lambda branch "
           f"{'on, shipped default' if cfg.ECQCO_LAMB else 'off, output-preserving'})", "-")
    if not cfg.ECQCO_PY.exists():
        log.info(f"skipped      : {cfg.ECQCO_PY} not found")
        return None
    qr, cr = QuantumRegister(n, "q"), ClassicalRegister(n, "c")
    src = QuantumCircuit(qr, cr)
    src.compose(base, qubits=range(n), inplace=True)
    src.measure(qr, cr)
    work.mkdir(parents=True, exist_ok=True)
    nm = base.name
    (work / f"{nm}.qasm").write_text(qasm2.dumps(src))
    (work / f"list_{nm}.json").write_text(json.dumps(
        [{"name": nm, "qasm": str(work / f"{nm}.qasm"), "qubits": n}]))
    t = time.time()
    p = subprocess.run([str(cfg.ECQCO_PY), str(cfg.ECQCO_WRAPPERS / "run_ecqco_any.py"),
                        "--list", str(work / f"list_{nm}.json"), "--out", str(work),
                        "--seed", str(cfg.SEED), "--lamb", "true" if cfg.ECQCO_LAMB else "false"],
                       capture_output=True, text=True)
    t_compile = time.time() - t
    for line in p.stdout.strip().splitlines()[-2:]:
        if line.strip():
            log.info(f"ecqco-env    : {line}")
    out = work / nm
    if p.returncode or not out.exists():
        log.info(f"FAILED       : {(p.stderr or p.stdout).strip()[-300:]}")
        return None
    obf = QuantumCircuit.from_qasm_file(str(out))
    log.info(f"compile      : {t_compile:.1f}s in ecqco-env (pyqpanda3 + mindquantum)")
    describe("ECQCO", obf)
    draw(unmeasured(obf), "ECQCO circuit")
    u = unmeasured(obf)
    rec = {"scheme": "ECQCO", "qubits": obf.num_qubits, "gates": u.size(), "depth": u.depth(),
           "twoq": n2q(u), "compile_s": t_compile}
    rec.update(structure(base, obf))
    rec["semantics"] = semantics(obf, n, ref, dev)
    rec.update(run_whole(dev, obf, n, ref))
    return rec
