# Manuscript figures

```bash
python paper_figs/make_all.py
```

This rebuilds the data table of every figure from the saved results and leaves the PDF and a 600-dpi PNG in each
figure's `out/` directory. Nothing here simulates or submits a QPU job. Figure 1 is drawn by
`fig_workflow/build.py` (editable PowerPoint); figure 2 by `scripts/dag_paper_style_hw_v4.py bell_n4`.

| Figure | Output | Input |
|---|---|---|
| 1 | `fig_workflow` | drawn by `build.py` (illustration, no data) |
| 2 | `fig_masking_cut_compiled_v4` | `results/hw_final_v4/ibm_marrakesh/` (submitted bell_n4 fragments); `out/numbers.json` |
| 3 | `fig_v4_normged_all` | `runs/tier1_tier2/metrics/structure_FakeMarrakesh/`; Tier 2, 23 circuits |
| 4 | `fig_ecqco4_normged` | `data/ecqco4.csv`: ECQCO's values from its paper; Q-MaskCut from `runs/ecqco4/` |
| 5 | `fig_v4_tvd_noiseless` | Tier 1 noiseless: `runs/tier1_tier2` (baselines), `runs/tier1_uniformkey` (Q-MaskCut) |
| 6 | `fig_v4_tvd_marrakesh` | Tier 1 on FakeMarrakesh, same folders |
| 7 | `fig_v4_bv3_metrics` | `runs/bv3/`; BV-3, 20 iterations x 10 inputs |
| 8 | `fig_v4_bv3_bitstrings` | `runs/bv3/` |
| 9 | `fig_v4_fidelity_noisy` | Tier 1 on FakeMarrakesh and FakeMontrealV2, same folders as figure 5 |
| 10 | `fig_v4_time_all` | `runs/tier1_tier2/metrics/structure_FakeMarrakesh/` |
| 11 | `fig_v4_depth_all` | `runs/tier1_tier2/metrics/structure_FakeMarrakesh/` |
| 12 | `fig_hw_fidelity` | `data/hw_fidelity.csv` from `results/hw_final_v4/ibm_marrakesh/results.csv` |

Each figure's table is in its `data/` directory. All circuits are compiled at `optimization_level=3` (best of 8
layouts on good qubits); normGED follows ECQCO's Eq. 14 on the compiled circuits. The shared style is in
`qmaskcut_style.py`. `make_v4.py` also writes a few supplementary figures that are not in the manuscript.
