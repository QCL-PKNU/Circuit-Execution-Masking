"""Regenerate the manuscript figures 3-12 from the saved results (no simulation, no QPU job)."""
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SCRIPTS = ["make_v4.py",                          # figures 3, 5-11
           "fig_ecqco4_normged/plot.py",          # figure 4
           "fig_hw_fidelity/plot.py"]             # figure 12


def main():
    for script in SCRIPTS:
        result = subprocess.run([sys.executable, str(HERE / script)], capture_output=True, text=True)
        print(result.stdout.strip())
        if result.returncode:
            print(result.stderr, file=sys.stderr)
            return result.returncode
    print("All manuscript figures built.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
