"""Q-MaskCut: protect a quantum circuit, cut it with CutQC, run the fragments, decode.

    masking/      decoy wires, idle-time padding, CZ padlocks, QOTP twirl
    cutting/      CutQC cut search, fragment instances, reconstruction, window marginals
    metrics/      ideal distributions, fidelity and TVD, normGED and GED_2Q
    device/       execution on a fake IBM backend
    baselines/    ECQCO and ObfusQate, for comparison
    experiments/  the per-circuit pipeline (cut study + Q-MaskCut) and the report
    utils/        circuit helpers and the step-by-step log
    config.py     run settings, overwritten by main.py
"""
