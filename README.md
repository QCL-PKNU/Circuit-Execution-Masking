# Circuit-Execution-Masking: Evaluating Quantum Circuit Protection Methods After Compilation

Measure how much a quantum circuit protection method still hides once the circuit is compiled. Every protected
circuit is compiled at optimisation level 3 before its structure (normGED) and output (TVD) are measured, and again
after the provider recompiles it. The comparison covers ObfusQate's inverse, composite and delayed gates, ECQCO, and
Q-MaskCut, one example combination of existing techniques: a one-time-pad key, decoy wires with CZ padlocks, CutQC
wire cutting with a random rotation pair across each cut, and Pauli twirling with no-pattern dynamical decoupling
after compilation.

## Try Q-MaskCut

Python 3.12 and a Gurobi licence (`gurobi.lic` in this directory; free for academic use) are needed to cut circuits.
From this directory:

```
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python qmaskcut_eval.py --run runs/try --circuits bv_n3 bell_n4 --noise none \
  --key-mode uniform --methods original Q-MaskCut
```

Expected summary (noiseless, 8192 shots per fragment variant):

```
bell_n4  original   fid 0.9994 | TVD_obf_min 0.019 | TVD_deobf 0.019 | normGED 0.000 | depth 13
bell_n4  Q-MaskCut  fid 0.9995 | TVD_obf_min 0.252 | TVD_deobf 0.014 | normGED 0.784 | depth 25
bv_n3    original   fid 1.0000 | TVD_obf_min 0.000 | TVD_deobf 0.000 | normGED 0.000 | depth 7
bv_n3    Q-MaskCut  fid 0.9997 | TVD_obf_min 1.000 | TVD_deobf 0.000 | normGED 0.832 | depth 19
```

`TVD_obf_min` is the distance between the correct output and what the provider sees without the key (high: hidden);
`TVD_deobf` is the distance after decoding with the key (low: recovered). These values equal the saved results in
`runs/tier1_uniformkey` and `runs/tier1_tier2`. See [experiment instructions](src/qmaskcut/README.md) for every
script, run folder and option.

## Reproduce the experiments

```
python paper_figs/make_all.py
python tvd_key_mean.py --run runs/tier1_keymean
python scripts/hw_decode_offline.py
python recompile_check.py --run runs/check --circuits bv_n3 bell_n4
```

The first three commands use the supplied data only: they rebuild every manuscript figure, the TVD averaged over
keys, and the ibm_marrakesh results from the saved raw counts, without simulation, Gurobi or QPU jobs. The last one
repeats the second-compilation measurement of table 1 for two circuits. [Experiment instructions](src/qmaskcut/README.md#experiments)
describe the full runs and the hardware execution. [Figure inputs](paper_figs/README.md) identify the data used by each
manuscript plot.

## License

New source code is MIT; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Vendored CutQC retains its
[MIT license](CutQC/LICENSE); the QASMBench circuits retain their [license](QASMBench/LICENSE).
