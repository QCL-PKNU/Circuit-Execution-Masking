"""Run settings shared by every module. main.py overwrites them from its SETTINGS block."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QASMBENCH = ROOT / "QASMBench"
ECQCO_DIR = ROOT / "external_benchmarks" / "ecqco"
ECQCO_PY = ROOT / "ecqco-env" / "bin" / "python"
# our wrappers around ECQCO's unmodified QCOO/QCSO (they live in this repo, not in ECQCO's)
ECQCO_WRAPPERS = ROOT / "src" / "qmaskcut" / "baselines" / "ecqco_wrappers"
OBF_DIR = (ROOT / "external_benchmarks" / "obfusqate" / "Obfuscation Techniques"
           / "Circuit Obfuscation Algorithms")

CIRCUITS: list = []
ONLY: list = []
BACKEND = "FakeMontreal"
SHOTS = 4096
OPT_LEVEL = 1
SEED = 42
MAX_CUTS = 2
RUN_CUT_STUDY = True
RUN_QMASKCUT = True
RUN_ECQCO = True
RUN_OBFUSQATE = True
QMC_DECOYS = 2
QMC_PADLOCKS = 2
ECQCO_LAMB = False
GED_METHOD = "deterministic"   # or "networkx" (time-limited search; not reproducible)
GED_SWEEPS = 40
GED_TIMEOUT = 20
GED_NX_UP_TO = 8
ASCII_MAX_OPS = 400
ASCII_FOLD = 150
MPS_ABOVE = 20
MAKE_FIGURES = True


def update(**settings):
    """Apply settings by name; an unknown name is a typo, so it is an error."""
    g = globals()
    for k, v in settings.items():
        if k not in g or k.startswith("_") or callable(g[k]):
            raise KeyError(f"unknown setting {k!r}")
        g[k] = v

# ObfusQate arms to run in main.py. Inverse gates make circuits 10-100x deeper; above about 18
# qubits a noisy simulation of them does not finish, so they can be left out there.
OBFUSQATE_ARMS = ["inverse gates", "composite gates", "delayed gates"]

# Circuits scored on the qubits their QASMBench file measures, not the whole output.
# kNN and swap test measure only the ancilla q0; their whole output is spread over
# millions of strings, which no 4096-shot run can score. The exact cut check still
# compares the FULL reconstruction.
SCORE_QUBITS = {"knn_n25": [0], "swap_test_n25": [0]}
