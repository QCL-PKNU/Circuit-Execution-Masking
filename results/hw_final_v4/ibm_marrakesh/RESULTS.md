# Hardware results: ibm_marrakesh, full method

449 circuits in three jobs (426 fragment variants + 23 originals), 2048 shots each, 260 s of device time.
Original: compiled at optimisation level 3 and run as it is. Q-MaskCut: fragments run on the device, rebuilt with CutQC,
decoys dropped and key removed. IBM runtime DD and twirling off for every circuit. Fidelity: Hellinger fidelity to the
exact output (ising_n34 and ising_n42: averaged over 12-qubit blocks against an exact MPS reference).
Every protected circuit passed the noiseless check (decoded fidelity 0.996 to 1.000) before submission.

Data: `results.csv` (this folder), `raw_counts.json` (every count), `jobs.json` (job IDs). Figure: `paper_figs/fig_hw_fidelity`.

| circuit | qubits | cuts | fragment variants | original | Q-MaskCut (with key) | Q-MaskCut without key | what it means |
|---|---|---|---|---|---|---|---|
| bv_n3 | 3 | 1 | 7 | 0.939 | 0.9332 | 0.0234 | Short circuit: both high, difference -0.006; without the key 0.023. |
| bv_n4 | 4 | 1 | 7 | 0.8628 | 0.9177 | 0 | Q-MaskCut higher by +0.055 (fragments about as long as the original: 2.98 against 2.96 us); without the key 0.000. |
| bell_n4 | 4 | 1 | 7 | 0.9914 | 0.9903 | 0.5689 | Short circuit: both near 1 (difference -0.001). Without the key 0.57, the highest a bit flip can hide bell_n4. |
| qpe_n9 | 9 | 2 | 24 | 0.5245 | 0.6834 | 0.01145 | Q-MaskCut higher by +0.159: its longest fragment runs 8.14 us against 10.78 us for the original; without the key 0.011. |
| adder_n10 | 10 | 2 | 24 | 0.3276 | 0.4987 | 0 | Q-MaskCut higher by +0.171: its longest fragment runs 8.87 us against 12.98 us for the original; without the key 0.000. |
| seca_n11 | 11 | 2 | 24 | 0.09255 | 0.2799 | 0 | Q-MaskCut higher by +0.187: its longest fragment runs 8.4 us against 12.04 us for the original; without the key 0.000. |
| multiply_n13 | 13 | 2 | 25 | 0.252 | 0.3215 | 0 | Q-MaskCut higher by +0.070: its longest fragment runs 7.03 us against 8.32 us for the original; without the key 0.000. |
| bv_n14 | 14 | 1 | 7 | 0.5242 | 0.2164 | 0.00035 | Q-MaskCut lower by -0.308 (under check); without the key 0.000. |
| qf21_n15 | 15 | 2 | 24 | 0.1054 | 0.4148 | 0.00044 | Q-MaskCut higher by +0.309: its longest fragment runs 22.14 us against 29.35 us for the original; without the key 0.000. |
| qec9xz_n17 | 17 | 2 | 24 | 0.1244 | 0.2611 | 0 | Q-MaskCut higher by +0.137: its longest fragment runs 6.01 us against 6.81 us for the original; without the key 0.000. |
| bv_n19 | 19 | 2 | 19 | 0.1016 | 0.1918 | 0 | Q-MaskCut higher by +0.090: its longest fragment runs 3.72 us against 8.68 us for the original; without the key 0.000. |
| cat_state_n22 | 22 | 2 | 19 | 0.2762 | 0.3664 | 0 | Q-MaskCut higher by +0.090: its longest fragment runs 3.62 us against 4.96 us for the original; without the key 0.000. |
| ghz_state_n23 | 23 | 2 | 19 | 0.2802 | 0.2382 | 0 | Q-MaskCut lower by -0.042 (under check); without the key 0.000. |
| knn_n25 | 25 | 1 | 7 | 0.00065 | 0.08546 | 0 | Q-MaskCut higher by +0.085: its longest fragment runs 14.88 us against 21.67 us for the original; without the key 0.000. |
| ising_n26 | 26 | 1 | 7 | 3e-05 | 0.01922 | 0.01922 | Uniform output: a noise-free device would also score 3e-05 with 2048 shots, which is what the original got. Q-MaskCut is higher only because it rebuilds from several fragment runs; not counted as a gain. |
| ising_n34 | 34 | 1 | 7 | 0.4726 | 0.7437 | 0.7446 | Uniform output: a noise-free device would also score 0.52 with 2048 shots, which is what the original got. Q-MaskCut is higher only because it rebuilds from several fragment runs; not counted as a gain. |
| ghz_n40 | 40 | 1 | 7 | 0.00195 | 0.02231 | 0 | Device limit: one or two correct bitstrings of 40 bits and a 420-layer original; both return almost no correct shot. |
| ising_n42 | 42 | 1 | 7 | 0.5195 | 0.611 | 0.6114 | Uniform output: a noise-free device would also score 0.53 with 2048 shots, which is what the original got. Q-MaskCut is higher only because it rebuilds from several fragment runs; not counted as a gain. |
| dj_n50 | 50 | 1 | 7 | 0 | 0.0001 | 0 | Device limit: one or two correct bitstrings of 50 bits and a 518-layer original; both return almost no correct shot. |
| cat_n65 | 65 | 2 | 19 | 0.0061 | 0.00136 | 0 | Device limit: one or two correct bitstrings of 65 bits and a 195-layer original; both return almost no correct shot. |
| dj_n70 | 70 | 3 | 61 | 0 | 0 | 0 | Device limit: one or two correct bitstrings of 70 bits and a 754-layer original; both return almost no correct shot. |
| ghz_n78 | 78 | 3 | 31 | 0.00171 | 0.00175 | 0 | Device limit: one or two correct bitstrings of 78 bits and a 234-layer original; both return almost no correct shot. |
| ghz_n127 | 127 | 4 | 43 | 0 | 9e-05 | 0 | Device limit: one or two correct bitstrings of 127 bits and a 1204-layer original; both return almost no correct shot. |

## Summary

- **Up to 25 qubits (14 circuits):** Q-MaskCut higher by more than 0.02 on 10, within 0.02 on bv_n3 and bell_n4, lower on bv_n14 and ghz_state_n23; mean 0.457 against 0.386.
- **Ising circuits (3):** uniform output, so the fidelity reflects the number of shots, not noise; not counted.
- **40 qubits and more, except ising_n42 (6):** both near 0; the device noise is too large at these sizes.
- **Without the key:** at most 0.023 on every circuit except bell_n4 (0.57, its ceiling) and the two block-scored Ising circuits (uniform blocks are unchanged by a bit flip).
