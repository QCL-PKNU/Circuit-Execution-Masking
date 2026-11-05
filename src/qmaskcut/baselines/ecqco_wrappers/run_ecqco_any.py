"""Run the ECQCO pipeline on ARBITRARY circuits, not just the four Fig5 ones.

    ENV: ecqco-env   (pyqpanda3 + mindquantum -- the README-pinned stack)
    RUN: ecqco-env/bin/python external_benchmarks/ecqco/repro/run_ecqco_any.py \
             --list circuits.json --out DIR [--seed 42]

`--list` is a JSON array of {"name":..., "qasm": <abs path>, "qubits": n}.
Writes one transformed OpenQASM per circuit for scoring in another environment.
"""
import os, sys, io, json, random, argparse, warnings, contextlib, traceback
warnings.filterwarnings("ignore")
ECQ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..",
                                   "external_benchmarks", "ecqco"))   # ECQCO, unmodified
os.chdir(ECQ); sys.path.insert(0, ECQ)
os.makedirs("result", exist_ok=True)
from QCOO import QCOOManager
from QCSO import QCSO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lamb", default="true", choices=["true","false"])
    ap.add_argument("--shots", type=int, default=1000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = json.load(open(a.list))
    lamb = (a.lamb == "true")
    made = []
    for j in jobs:
        nm, src, nq = j["name"], j["qasm"], j["qubits"]
        tag = os.path.join(a.out, nm)
        try:
            random.seed(a.seed)
            with contextlib.redirect_stdout(io.StringIO()):
                q = QCOOManager(nq, src, tag)
                qcoo_path, _ = q.QCOO(None)
                QCSO(qcoo_path, tag, lamb_param=lamb, total_shots=a.shots)
            made.append({"name": nm, "qubits": nq, "qcoo": qcoo_path, "tag": tag, "ok": True})
            print(f"ok    {nm:<18} {nq:>4}q", flush=True)
        except Exception as e:
            made.append({"name": nm, "qubits": nq, "ok": False,
                         "error": f"{type(e).__name__}: {str(e)[:120]}"})
            print(f"FAIL  {nm:<18} {nq:>4}q  {type(e).__name__}: {str(e)[:70]}", flush=True)
    json.dump(made, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    ok = sum(1 for m in made if m["ok"])
    print(f"\n{ok}/{len(made)} circuits transformed -> {a.out}")

if __name__ == "__main__":
    main()
