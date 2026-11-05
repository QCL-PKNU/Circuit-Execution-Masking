# Evaluation report

Run folder: `runs/tier1_tier2`

Mean over circuits per backend and method:

|                                                |   fidelity |   tvd_obf |   tvd_obf_min |   tvd_deobf |   tvd_ideal |   normged |   ged2q |    depth |   time_us |
|:-----------------------------------------------|-----------:|----------:|--------------:|------------:|------------:|----------:|--------:|---------:|----------:|
| ('FakeMarrakesh', 'ECQCO')                     |     0.2225 |    0.7755 |        0.7755 |      0.7755 |        0.7  |    0.4131 |  0.2979 |  200.3   |    9.8864 |
| ('FakeMarrakesh', 'Q-MaskCut')                 |     0.775  |    0.9688 |        0.9128 |      0.244  |        0    |    0.89   |  0.4465 |  249.182 |    7.136  |
| ('FakeMarrakesh', 'composite gates')           |     0.7555 |    0.2596 |        0.2596 |      0.2596 |        0    |    0.2313 |  0.0418 |  184.273 |    9.2898 |
| ('FakeMarrakesh', 'delayed gates')             |     0.7724 |    0.2427 |        0.2427 |      0.2427 |        0    |    0.2467 |  0.1365 |  186.182 |    9.4524 |
| ('FakeMarrakesh', 'inverse gates')             |     0.2094 |    0.8045 |        0.8045 |      0.8045 |        0    |    0.9682 |  0.9739 | 2953.12  |  111.249  |
| ('FakeMarrakesh', 'original')                  |     0.7661 |    0.2497 |        0.2497 |      0.2497 |        0    |    0      |  0      |  179.454 |    9.2495 |
| ('FakeMontrealV2', 'ECQCO')                    |     0.0913 |    0.9072 |        0.9072 |      0.9072 |        0.75 |    0.5518 |  0.3763 |  128.875 |   31.9874 |
| ('FakeMontrealV2', 'Q-MaskCut')                |     0.5327 |    0.9541 |        0.8911 |      0.4802 |        0    |    0.9404 |  0.444  |  420.333 |   21.9927 |
| ('FakeMontrealV2', 'composite gates')          |     0.5509 |    0.4625 |        0.4625 |      0.4625 |        0    |    0.2659 |  0.0458 |  152.778 |   30.3697 |
| ('FakeMontrealV2', 'delayed gates')            |     0.5521 |    0.4585 |        0.4585 |      0.4585 |        0    |    0.3356 |  0.091  |  151.778 |   31.125  |
| ('FakeMontrealV2', 'inverse gates')            |     0.1446 |    0.8756 |        0.8756 |      0.8756 |        0    |    0.9564 |  0.9729 | 1760.43  |  368.704  |
| ('FakeMontrealV2', 'original')                 |     0.545  |    0.4651 |        0.4651 |      0.4651 |        0    |    0      |  0      |  130.778 |   29.8781 |
| ('none', 'ECQCO')                              |     0.3    |    0.7014 |        0.7014 |      0.7014 |        0.7  |    0.4131 |  0.2979 |  200.3   |    9.8864 |
| ('none', 'Q-MaskCut')                          |     0.9894 |    0.9737 |        0.9159 |      0.0219 |        0    |    0.89   |  0.4465 |  249.182 |    7.136  |
| ('none', 'composite gates')                    |     0.9958 |    0.0147 |        0.0147 |      0.0147 |        0    |    0.2313 |  0.0418 |  184.273 |    9.2898 |
| ('none', 'delayed gates')                      |     0.9964 |    0.0147 |        0.0147 |      0.0147 |        0    |    0.2467 |  0.1365 |  186.182 |    9.4524 |
| ('none', 'inverse gates')                      |     0.9961 |    0.0145 |        0.0145 |      0.0145 |        0    |    0.9711 |  0.9759 | 3453.45  |  130.771  |
| ('none', 'original')                           |     0.9959 |    0.0168 |        0.0168 |      0.0168 |        0    |    0      |  0      |  179.454 |    9.2495 |
| ('structure_FakeMarrakesh', 'ECQCO')           |   nan      |  nan      |      nan      |    nan      |      nan    |    0.399  |  0.2908 |  179.5   |    9.1975 |
| ('structure_FakeMarrakesh', 'Q-MaskCut')       |   nan      |  nan      |      nan      |    nan      |      nan    |    0.8937 |  0.3489 |  245.375 |    7.2676 |
| ('structure_FakeMarrakesh', 'composite gates') |   nan      |  nan      |      nan      |    nan      |      nan    |    0.21   |  0.0486 |  372.812 |   17.4736 |
| ('structure_FakeMarrakesh', 'delayed gates')   |   nan      |  nan      |      nan      |    nan      |      nan    |    0.2202 |  0.0749 |  360.531 |   17.4721 |
| ('structure_FakeMarrakesh', 'inverse gates')   |   nan      |  nan      |      nan      |    nan      |      nan    |    0.9826 |  0.9864 | 9044.5   |  373.771  |
| ('structure_FakeMarrakesh', 'original')        |   nan      |  nan      |      nan      |    nan      |      nan    |    0      |  0      |  354.188 |   17.346  |

## Checks

- operator check False: FakeMarrakesh adder_n10 delayed gates
- operator check False: FakeMarrakesh bell_n4 delayed gates
- operator check False: FakeMarrakesh bv_n3 delayed gates
- operator check False: FakeMarrakesh bv_n4 delayed gates
- operator check False: FakeMarrakesh qpe_n9 delayed gates
- operator check False: FakeMontrealV2 adder_n10 delayed gates
- operator check False: FakeMontrealV2 bell_n4 delayed gates
- operator check False: FakeMontrealV2 bv_n3 delayed gates
- operator check False: FakeMontrealV2 bv_n4 delayed gates
- operator check False: FakeMontrealV2 qpe_n9 delayed gates
- operator check False: none adder_n10 delayed gates
- operator check False: none bell_n4 delayed gates
- operator check False: none bv_n3 delayed gates
- operator check False: none bv_n4 delayed gates
- operator check False: none qpe_n9 delayed gates
- FakeMarrakesh bell_n4 ECQCO: N/A (ECQCO's code fails on this circuit)
- FakeMarrakesh bv_n19 inverse gates: not simulated: too deep for noisy simulation (13719 gates x 2^21 = 2.9e+10 per shot > 3e+08)
- FakeMarrakesh multiply_n13 inverse gates: not simulated: too deep for noisy simulation (10321 gates x 2^17 = 1.4e+09 per shot > 3e+08)
- FakeMarrakesh qec9xz_n17 inverse gates: not simulated: too deep for noisy simulation (11777 gates x 2^17 = 1.5e+09 per shot > 3e+08)
- FakeMontrealV2 bell_n4 ECQCO: N/A (ECQCO's code fails on this circuit)
- FakeMontrealV2 bv_n14 Q-MaskCut: FAIL FileExistsError: runs/tier1_tier2/circuits/FakeMontrealV2/bv_n14/original__run_sched.qasm exists with different content (RunStore never overwrites)
- FakeMontrealV2 bv_n19 inverse gates: not simulated: too deep for noisy simulation (7267 gates x 2^19 = 3.8e+09 per shot > 3e+08)
- FakeMontrealV2 multiply_n13 Q-MaskCut: FAIL FileExistsError: runs/tier1_tier2/circuits/FakeMontrealV2/multiply_n13/original__run_sched.qasm exists with different content (RunStore never overwrites)
- FakeMontrealV2 qec9xz_n17 inverse gates: not simulated: too deep for noisy simulation (6669 gates x 2^17 = 8.7e+08 per shot > 3e+08)
- none bell_n4 ECQCO: N/A (ECQCO's code fails on this circuit)
- structure_FakeMarrakesh adder_n118 original: not computed: normGED on the compiled circuit did not finish (10 h with inverse gates, 50 min without)
- structure_FakeMarrakesh adder_n118 Q-MaskCut: not computed: normGED on the compiled circuit did not finish (10 h with inverse gates, 50 min without)
- structure_FakeMarrakesh adder_n118 inverse gates: not computed: normGED on the compiled circuit did not finish (10 h with inverse gates, 50 min without)
- structure_FakeMarrakesh adder_n118 composite gates: not computed: normGED on the compiled circuit did not finish (10 h with inverse gates, 50 min without)
- structure_FakeMarrakesh adder_n118 delayed gates: not computed: normGED on the compiled circuit did not finish (10 h with inverse gates, 50 min without)
- structure_FakeMarrakesh adder_n118 ECQCO: not computed: normGED on the compiled circuit did not finish (10 h with inverse gates, 50 min without)
- structure_FakeMarrakesh adder_n28 ECQCO: N/A (no ECQCO output for this circuit)
- structure_FakeMarrakesh adder_n64 original: not computed: normGED on the compiled circuit did not finish (10 h with inverse gates, 50 min without)
- structure_FakeMarrakesh adder_n64 Q-MaskCut: not computed: normGED on the compiled circuit did not finish (10 h with inverse gates, 50 min without)
- structure_FakeMarrakesh adder_n64 inverse gates: not computed: normGED on the compiled circuit did not finish (10 h with inverse gates, 50 min without)
- structure_FakeMarrakesh adder_n64 composite gates: not computed: normGED on the compiled circuit did not finish (10 h with inverse gates, 50 min without)
- structure_FakeMarrakesh adder_n64 delayed gates: not computed: normGED on the compiled circuit did not finish (10 h with inverse gates, 50 min without)
- structure_FakeMarrakesh adder_n64 ECQCO: not computed: normGED on the compiled circuit did not finish (10 h with inverse gates, 50 min without)
- structure_FakeMarrakesh bell_n4 ECQCO: N/A (no ECQCO output for this circuit)
- structure_FakeMarrakesh bv_n140 ECQCO: N/A (no ECQCO output for this circuit)
- structure_FakeMarrakesh bv_n70 ECQCO: N/A (no ECQCO output for this circuit)
- structure_FakeMarrakesh cat_n130 ECQCO: N/A (no ECQCO output for this circuit)
- structure_FakeMarrakesh cat_n35 ECQCO: N/A (no ECQCO output for this circuit)
- structure_FakeMarrakesh cat_n65 ECQCO: N/A (no ECQCO output for this circuit)
- structure_FakeMarrakesh dj_n50 ECQCO: N/A (no ECQCO output for this circuit)
- structure_FakeMarrakesh dj_n70 ECQCO: N/A (no ECQCO output for this circuit)
- structure_FakeMarrakesh ghz_n127 ECQCO: N/A (no ECQCO output for this circuit)
- structure_FakeMarrakesh ghz_n40 ECQCO: N/A (no ECQCO output for this circuit)
- structure_FakeMarrakesh ghz_n78 ECQCO: N/A (no ECQCO output for this circuit)
- structure_FakeMarrakesh ising_n34 ECQCO: N/A (no ECQCO output for this circuit)
- structure_FakeMarrakesh ising_n42 ECQCO: N/A (no ECQCO output for this circuit)
- structure_FakeMarrakesh ising_n66 ECQCO: N/A (no ECQCO output for this circuit)
- structure_FakeMarrakesh knn_n31 ECQCO: N/A (no ECQCO output for this circuit)
