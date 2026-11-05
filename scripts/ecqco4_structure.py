"""Q-MaskCut normGED on ECQCO's four 7-qubit benchmark circuits with the full method (paper figure: ECQCO benchmarks), through tier2_structure.py
(same code as Tier 2: FakeMarrakesh, level 3 best of 8, good qubits, twirl + no-pattern DD). Output:
runs/ecqco4/metrics/structure_FakeMarrakesh/<name>.json"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]
import qmaskcut_eval
FILES = {"ecqco_bv_n7": "Bernstein-Vazirani/BV-7.qasm", "ecqco_grover_n7": "Grover/Grover-7.qasm", "ecqco_qaoa_n7": "QAOA/QAOA-7.qasm",
         "ecqco_shor_n7": "Shor/shor-7.qasm"}
_orig = qmaskcut_eval.circuit_path
qmaskcut_eval.circuit_path = lambda name: ROOT / "qasm" / FILES[name] if name in FILES else _orig(name)
import tier2_structure
sys.argv = ["tier2_structure.py", "--run", "runs/ecqco4", "--circuits", *FILES, "--workers", "4",
            "--methods", "original", "Q-MaskCut"]
tier2_structure.main()
