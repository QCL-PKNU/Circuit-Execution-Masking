"""Fig hardware fidelity: Hellinger fidelity to the exact output on ibm_marrakesh, 2048 shots, 23 circuits (full-method run,
data from results/hw_final_v4/ibm_marrakesh/results.csv). Original
compiled at optimization_level=3 and submitted as it is; Q-MaskCut rebuilt and decoded with the key."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import pandas as pd
from qmaskcut_style import (house_figure, grouped_bars, finish, save_fig, check_overlaps, LABEL_FS, NA_LABEL, BASE)

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
name = lambda c: c + (" (MQT)" if str(c).startswith("dj_") else "")

df = pd.read_csv(f"{D}/hw_fidelity.csv")
S = ["Original", "Q-MaskCut"]
fig, ax, s = house_figure()
x = grouped_bars(ax, s, S, {k: df[k].tolist() for k in S}, log=True,
                 fmt=lambda v: "0" if v == 0 else (f"{v:.2f}" if v >= 0.01 else f"{v:.1e}"), floor=1e-4, label_fs=LABEL_FS, na=NA_LABEL)
finish(fig, ax, s, x, [f"{name(c)} ({n})" for c, n in zip(df.circuit, df.n)],
       "fidelity (log scale)", (1e-4 / 1.5, 12), [10.0 ** k for k in range(-4, 1)], ncol=2)   # no tick above 1
save_fig(fig, OUT, "fig_hw_fidelity")
check_overlaps(fig)
