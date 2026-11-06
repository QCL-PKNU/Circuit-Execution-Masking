# Experiment instructions

Paths and commands are relative to the repository root.

## Contents

| Path | What |
|---|---|
| `src/qmaskcut/` | the Q-MaskCut package: protection (key, decoys, padlocks), cutting (CutQC interface), masking, metrics (normGED, TVD, fidelity) |
| `CutQC/` | CutQC (Tang *et al.*, MIT licence), vendored with the changes in `patches/cutqc.patch` |
| `qmaskcut_eval.py` | Tier 1: 11 circuits simulated end to end, without noise and on FakeMarrakesh / FakeMontrealV2 |
| `tier2_structure.py` | Tier 2: compiled structure (normGED, depth, time) of 23 circuits of 3 to 127 qubits on FakeMarrakesh |
| `recompile_check.py` | second compilation by the provider (tables 1 and 2) |
| `trackb_bv3.py`, `trackb_compiled_normged.py`, `trackb_figures.py` | BV-3 study, 20 iterations x 10 random inputs (figures 7 and 8) |
| `tvd_key_mean.py` | TVD without the key averaged over all keys (section 4.4) |
| `scripts/hw_final_v4.py` | hardware run on ibm_marrakesh (prepare, submit, collect) |
| `scripts/hw_decode_offline.py` | rebuilds the hardware results from the saved raw counts, without an IBM account |
| `scripts/ecqco4_structure.py` | Q-MaskCut on ECQCO's four benchmark circuits (figure 4) |
| `scripts/dag_paper_style_hw_v4.py` | gate-dependency graphs of bell_n4 (figure 2) |
| `paper_figs/` | figure scripts, the data table of every figure (`data/`) and the figures (`out/`) |
| `runs/`, `results/` | all results the paper reads (see below) |
| `QASMBench/`, `MQTBench_*/`, `data/circuits/`, `qasm/` | the benchmark circuits used (QASMBench under its BSD licence) |

## Result folders

| Folder | Paper |
|---|---|
| `runs/tier1_tier2/` | Tier 1 for the original and the baselines; Tier 2 structure (figures 3, 10, 11) |
| `runs/tier1_uniformkey/` | Tier 1 for Q-MaskCut with the uniform random key (figures 5, 6, 9) |
| `runs/tier1_keymean/` | decoded outputs of that run and the TVD averaged over keys (section 4.4) |
| `runs/bv3/` | BV-3 study (figures 7, 8) |
| `runs/recompile/` | second compilation: Q-MaskCut on Tier 2 and BV-3, and every method (tables 1, 2) |
| `runs/ecqco4/` | ECQCO's benchmark circuits (figure 4) |
| `results/hw_final_v4/ibm_marrakesh/` | hardware run: job IDs, raw counts of all 449 submitted circuits, the submitted circuits (QPY), results (figure 12, tables 3, 4) |
| `docs/ecqco_outputs/` | the circuits ECQCO's implementation produced in our runs |

Each run folder holds `config.json` (the settings), `metrics/` (one JSON file per circuit and method), and where
simulated, `counts/` (every simulated count) and `circuits/` (the compiled circuits as QPY).

## Setup

Python 3.12:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

**Gurobi licence.** CutQC's cut search uses Gurobi. Put a Gurobi WLS licence file at `gurobi.lic` in the repository
root (`.gitignore` keeps it out of git). It is needed only to cut circuits again; the figures, tables and the offline
hardware decoding do not need it.

**Baselines.** The baselines' code is not redistributed here. To regenerate the baseline circuits, place ObfusQate's
released implementation under `external_benchmarks/obfusqate/` and ECQCO's implementation under
`external_benchmarks/ecqco/` (with its own environment `ecqco-env/`, `requirements-ecqco.txt`); see
`src/qmaskcut/config.py` for the exact paths. The circuits and results they produced in our runs are included. Without their code, run Q-MaskCut and the
original only: `--methods original Q-MaskCut` (`qmaskcut_eval.py`, `tier2_structure.py`).

## Package versions and seeds

Package versions are pinned in `requirements.txt` (Qiskit 2.5.1, Qiskit Aer 0.17.2, qiskit-ibm-runtime 0.48.0);
CutQC is vendored in `CutQC/` with the changes in `patches/cutqc.patch`. Every result uses these seeds:

| What | Seed | Where |
|---|---|---|
| Global seed | 42 | `SEED` in `src/qmaskcut/config.py` |
| Compilation (Tier 1, Tier 2, hardware) | optimisation level 3; best of 8 compilations with `seed_transpiler` 42 to 49, by estimated success probability; every variant of a fragment reuses that layout with seed 42 | `best()` in `scripts/new_workflow_test.py`, `--best-of 8` |
| Key, decoys and padlocks (Tier 1, Tier 2) | 42 (`--seed`); random key = the first draw of this seed | `protect_general(seed=...)`, `--key-mode uniform` |
| Rotation pair across each cut | `random.Random(1000 + seed)`, angle uniform in [0, 2π) | `qmaskcut_eval.py`, `tier2_structure.py`, `recompile_check.py` |
| Masking after compilation | 42 for the whole circuit, 42 + j for fragment variant j | `mask(..., seed)` |
| Aer simulator (Tier 1) | `seed × 10^7 + role × 10^5 + 1`, one per run and fragment variant, spaced so that no two runs share shots | `seedb()` in `qmaskcut_eval.py`; each seed is saved with its counts |
| BV-3 study | `seed_transpiler` 42; input state seed 1000 × iteration + input; protection seed 100 × iteration | `trackb_bv3.py` |
| Second compilation | `seed_transpiler` 42 | `recompile_check.py` |
| GED upper bound (local search) | 42 | `src/qmaskcut/metrics/normged.py` |
| Keys sampled for the key-averaged TVD (17 and 19 wires) | `numpy.random.default_rng(0)`, 20,000 keys | `tvd_key_mean.py` |

## Experiments

The figures and tables are rebuilt from the saved results in a few minutes, without simulation, Gurobi or device time:

```bash
.venv/bin/python paper_figs/make_all.py                    # figures 3-12
.venv/bin/python paper_figs/fig_workflow/build.py paper_figs/fig_workflow/out/fig_workflow.pptx   # figure 1 (editable)
.venv/bin/python tvd_key_mean.py --run runs/tier1_keymean            # section 4.4
.venv/bin/python scripts/hw_decode_offline.py              # hardware results (tables 3, 4) from the raw counts
```

Figure 2 (`scripts/dag_paper_style_hw_v4.py bell_n4`) rebuilds the hardware pipeline for bell_n4 and checks it against
the submitted fragments; it reads ibm_marrakesh's calibration of the run date, so it needs an IBM Quantum account
(read only, no job is submitted) and the Gurobi licence.

Running the experiments again (hours to days; the noisy simulations of Tier 1 take about 9 hours per noise model):

```bash
C="bv_n3 bv_n4 bell_n4 qpe_n9 adder_n10 seca_n11 multiply_n13 bv_n14 qf21_n15 qec9xz_n17 bv_n19"
for B in none FakeMarrakesh FakeMontrealV2; do
  .venv/bin/python qmaskcut_eval.py --run runs/new_tier1 --circuits $C --noise $B --key-mode uniform
done
.venv/bin/python tier2_structure.py --run runs/new_tier2
.venv/bin/python recompile_check.py --run runs/new_recompile               # table 1, Tier 2
.venv/bin/python recompile_check.py --run runs/new_recompile --bv3         # table 1, BV-3
.venv/bin/python recompile_check.py --run runs/new_recompile --baselines   # table 2
.venv/bin/python trackb_bv3.py --run runs/new_bv3 --key-mode uniform
.venv/bin/python trackb_compiled_normged.py --run runs/new_bv3
.venv/bin/python scripts/ecqco4_structure.py
```

A run folder can be re-analysed from its saved counts without simulating again: `qmaskcut_eval.py` reuses every count
already in `<run>/counts/`. For example, `runs/tier1_keymean/` was produced from the counts of
`runs/tier1_uniformkey/` (copy its `counts/` and `circuits/` folders there first).

The hardware run (`scripts/hw_final_v4.py --submit`) needs an IBM Quantum account and device time; its results can be
checked without one with `scripts/hw_decode_offline.py`, which decodes the saved raw counts with the same code.

