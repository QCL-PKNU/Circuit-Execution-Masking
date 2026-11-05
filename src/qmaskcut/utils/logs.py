"""Step-by-step run log: banners, ASCII circuit drawings, one-line circuit summaries."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from qiskit import QuantumCircuit

from .. import config as cfg
from ..utils.circuits import n2q, unmeasured

log = logging.getLogger("qmaskcut")

def setup_logging(out_dir: Path) -> Path:
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    path = out_dir / "run.log"
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(path)):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.setLevel(logging.INFO)
    log.propagate = False
    logging.getLogger().setLevel(logging.WARNING)       # keep library INFO quiet
    return path


def banner(text: str, ch: str = "="):
    log.info("")
    log.info(ch * 78)
    log.info(text)
    log.info(ch * 78)


def block(text: str):
    for line in text.splitlines():
        log.info("    " + line)


def draw(qc: QuantumCircuit, title: str):
    if qc.size() <= cfg.ASCII_MAX_OPS:
        log.info(f"{title}:")
        block(str(qc.draw(output="text", fold=cfg.ASCII_FOLD, idle_wires=True)))
    else:
        log.info(f"{title}: {qc.size()} operations, too many to draw "
                 f"(cfg.ASCII_MAX_OPS = {cfg.ASCII_MAX_OPS})")


def describe(label: str, qc: QuantumCircuit):
    q = unmeasured(qc)
    log.info(f"{label:12s} : {q.num_qubits} qubits, {q.size()} gates, depth {q.depth()}, "
             f"{n2q(q)} two-qubit")
