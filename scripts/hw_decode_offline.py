"""Rebuild the ibm_marrakesh results (results.csv) from the saved raw IBM counts, without an IBM account.

Inputs in results/hw_final_v4/ibm_marrakesh/: raw_counts.json (counts of every submitted circuit, as returned by
the device), groups.pkl (per circuit: the cut, fragment labels, key and exact reference) and
submitted_circuits.qpy.gz (the 449 circuits as submitted). The decoding is the same code as `hw_final_v4.py --collect`.

Usage: python scripts/hw_decode_offline.py [--out results_offline.csv]
"""
import argparse, csv, gzip, json, math, pickle, statistics as st, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]
from qiskit import qpy
from qmaskcut.cutting import cutqc as C
from qmaskcut.metrics.scoring import counts_to_le
from hw_final_v4 import decoded, hel, marginal
from hw_prepare import reference
from qmaskcut.utils.circuits import load

D = ROOT / "results/hw_final_v4/ibm_marrakesh"
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", default=str(D / "results_offline.csv"))
a = ap.parse_args()
res = {int(k): v for k, v in json.loads((D / "raw_counts.json").read_text()).items()}
groups = pickle.load(open(D / "groups.pkl", "rb"))
with gzip.open(D / "submitted_circuits.qpy.gz", "rb") as f:
    circuits = qpy.load(f)
rows = []
for g in groups:
    ref, n = g["ref"], g["n"]
    if "recompute" in ref:                     # ising_n26: 2^26 exact probabilities are recomputed, not stored
        ref, _ = reference(g["circuit"], load(g["circuit"], ref["recompute"]))
    po = counts_to_le(res[g["orig"]], n)
    if "windows" in ref:
        f_orig = st.mean(hel(marginal(po, w), r) for w, r in ref["windows"])
    else:
        f_orig = sum(math.sqrt(po.get(k, 0) * r) for k, r in ref.items()) ** 2
    cq = C._finish(g["lab"], g["subs"], g["cpm"])
    vecs = [C.counts_to_probs(res[i], circuits[i].num_clbits) for i in g["frag"]]
    C.evaluate_from(cq, dict(zip(g["labels"], vecs)))
    f, nk = decoded(cq, g["info"], n, ref, g["N"])
    rows.append({k: g[k] for k in ("circuit", "n", "K", "widths", "instances", "layers_original", "layers_fragment",
                                   "time_original_us", "time_fragment_us", "mask_pulses", "wires_masked",
                                   "normGED_fragment", "noiseless")} |
                {"fid_original": round(f_orig, 5), "fid_qmaskcut": round(f, 5), "fid_without_key": round(nk, 5)})
    print(g["circuit"], rows[-1]["fid_original"], rows[-1]["fid_qmaskcut"], flush=True)
with open(a.out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("saved", a.out)
