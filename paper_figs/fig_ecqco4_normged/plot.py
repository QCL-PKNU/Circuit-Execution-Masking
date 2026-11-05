"""Fig ECQCO benchmarks: normGED on ECQCO's four benchmark circuits. ECQCO and the ObfusQate schemes from
ECQCO's paper; Q-MaskCut: full method, measured after compilation with tier2_structure.py (FakeMarrakesh, level 3),
runs/ecqco4."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import pandas as pd
from qmaskcut_style import (house_figure, grouped_bars, finish, save_fig, check_overlaps, LABEL_FS, NA_LABEL, BASE)

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
name = lambda c: c + (" (MQT)" if str(c).startswith("dj_") else "")

df = pd.read_csv(f"{D}/ecqco4.csv")
S = ["ECQCO", "Inverse gates", "Delayed gates", "Composite gates", "Q-MaskCut"]
col = {k: (k if k == "Q-MaskCut" else f"{k} (reported)") for k in S}
fig, ax, s = house_figure()
x = grouped_bars(ax, s, S, {k: df[col[k]].tolist() for k in S}, fmt="{:.3f}", label_fs=LABEL_FS, na=NA_LABEL)
finish(fig, ax, s, x, ["Bernstein-Vazirani", "Grover", "QAOA", "Shor"], "normGED", (0, 1.12),
       np.arange(0, 1.01, 0.2), ncol=5, xlabel="ECQCO's benchmark circuit", rotation=0)
save_fig(fig, OUT, "fig_ecqco4_normged")
check_overlaps(fig)
