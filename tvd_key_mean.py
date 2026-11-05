"""TVD_obf averaged over keys (section 4.4).

The one-time-pad key is an XOR on the output bits, so the output the provider sees under key k is the decoded output
with its bits flipped by k: view_k[x] = dec[x ^ k]. TVD_obf(k) = TVD(view_k, ideal) therefore follows from the saved
decoded output of one run for every key, without new simulations. Keys are drawn as in algorithm 1 (each data wire with
probability 1/2, at least one), i.e. uniformly over the 2^n - 1 non-zero masks: all of them for n <= 15, else a sample.

Input: <run>/dists/<backend>/<circuit>.npz written by qmaskcut_eval.py (dec, ideal, nok, key).
Usage: python tvd_key_mean.py --run runs/tier1_keymean [--sample 20000]
"""
import argparse, glob, json
from pathlib import Path
import numpy as np

ORDER = ["bv_n3", "bv_n4", "bell_n4", "qpe_n9", "adder_n10", "seca_n11", "multiply_n13", "bv_n14", "qf21_n15",
         "qec9xz_n17", "bv_n19"]


def key_mask(dec, nok, key):
    """Bit mask of the drawn key in the array's index order, checked against the saved no-key view."""
    n = len(key); x = np.arange(len(dec))
    for m in (int(key, 2), int(key[::-1], 2)):
        if np.allclose(dec[x ^ m], nok, atol=1e-12):
            return m
    raise ValueError("key does not map the decoded output onto the no-key view")


def tvd_per_key(dec, ideal, keys, chunk=1 << 26):
    x = np.arange(len(dec)); out = np.empty(len(keys)); step = max(1, chunk // len(dec))
    for i in range(0, len(keys), step):
        k = keys[i:i + step]
        out[i:i + step] = 0.5 * np.abs(dec[x[None, :] ^ k[:, None]] - ideal[None, :]).sum(axis=1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True); ap.add_argument("--sample", type=int, default=20000)
    ap.add_argument("--exact-max", type=int, default=15, help="enumerate every key up to this many data wires")
    A = ap.parse_args()
    rng = np.random.default_rng(0); res = {}
    for d in sorted(glob.glob(f"{A.run}/dists/*")):
        B = Path(d).name; res[B] = {}
        for name in ORDER:
            f = Path(d) / f"{name}.npz"
            if not f.exists():
                continue
            z = np.load(f); dec, ideal, nok, key = z["dec"], z["ideal"], z["nok"], str(z["key"])
            n = len(key); m = key_mask(dec, nok, key)
            keys = (np.arange(1, 2 ** n) if n <= A.exact_max else rng.integers(1, 2 ** n, A.sample))
            t = tvd_per_key(dec, ideal, keys)
            res[B][name] = dict(n=n, keys=int(len(keys)), exact=bool(n <= A.exact_max), mean=float(t.mean()),
                                std=float(t.std()), min=float(t.min()), max=float(t.max()),
                                frac_ge_0_99=float((t >= 0.99).mean()), drawn=float(tvd_per_key(dec, ideal, np.array([m]))[0]))
            r = res[B][name]
            print(f"{B:15s} {name:13s} n={n:2d} keys={r['keys']:6d} mean {r['mean']:.3f} +/- {r['std']:.3f} "
                  f"[{r['min']:.3f}, {r['max']:.3f}]  drawn {r['drawn']:.3f}")
        if res[B]:
            print(f"{B:15s} MEAN over circuits: {np.mean([r['mean'] for r in res[B].values()]):.3f} "
                  f"(drawn keys {np.mean([r['drawn'] for r in res[B].values()]):.3f})")
    Path(A.run, "tvd_key_mean.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
