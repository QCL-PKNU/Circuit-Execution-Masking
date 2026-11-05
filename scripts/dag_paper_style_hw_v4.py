#!/usr/bin/env python3
"""The paper's worked-example DAG figure (same drawing style, colours, legend and EXPAND/REAL_GATES behaviour as
scripts/dag_paper_style_hw.py) for the v4 hardware run on ibm_marrakesh (results/hw_final_v4/ibm_marrakesh/prepared.pkl,
built by scripts/hw_final_v4.py, account pknu4):
  (a) the original as compiled and submitted,
  (b) the whole protected circuit before cutting (key, 2 decoys + 2 CZ padlocks, H on each decoy), compiled
      (level 3, best of 8 on the good-qubit map) + after-compile time-aware twirl + no-pattern DD (max 8 pairs per gap),
  (c, d) for each fragment, the variant that runs longest, exactly as submitted.
The device target is the one in force when the run was prepared (backend.target_history(2026-10-26 12:40 KST),
read only, pknu4); the script checks that it reproduces the stored original bit for bit, and rebuilds the whole
v4 pipeline for the circuit (key search, protection, cut, RX pair, fragment compile, twirl, DD) and asserts it
equals the stored fragments. No QPU job is submitted.

    .venv/bin/python scripts/dag_paper_style_hw_v4.py bell_n4
      -> docs/paper/physica_scripta/figs/fig_masking_cut_compiled_v4.pdf/.png
         (+ paper_figs/fig_masking_cut_compiled_v4/out/, results/new_workflow/dag_<name>_v4_panel_*.png)
"""
import datetime, itertools, json, math, os, pickle, random, subprocess, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from PIL import Image
import new_workflow_test as W
W.BEST_OF = 8                                   # as hw_final_v4.py
from qiskit import QuantumCircuit
from qiskit.circuit.library import HGate, RXGate
from qiskit.quantum_info import Statevector
from qmaskcut import config as cfg
from qmaskcut.cutting import cutqc as C
from qmaskcut.masking import protect_general
from qmaskcut.metrics.normged import deterministic_normged as ng
from qmaskcut.utils.circuits import load
from qmaskcut_eval import good_map
import hw_final_v4 as V4                        # CIRCS, MAX_PAIRS, PADLOCKS (its main block does not run on import)

NAME = sys.argv[1] if len(sys.argv) > 1 else "bell_n4"
EXPAND = os.environ.get("EXPAND", "1") == "1"   # draw every DD pulse as its own node
REAL = EXPAND and os.environ.get("REAL_GATES", "1") == "1"   # and show its real native gates rz, x, rz
RUN_PKL = ROOT / "results/hw_final_v4/ibm_marrakesh/prepared.pkl"
CACHE = Path(os.environ.get("V4_CACHE", tempfile.gettempdir())) / f"hw_final_v4_marrakesh_{NAME}.pkl"   # this circuit's slice of the 3.2 GB pickle
PREP_TIME = datetime.datetime(2026, 10, 26, 12, 40, tzinfo=datetime.timezone(datetime.timedelta(hours=9)))
OUT = ROOT / "results/new_workflow"; OUT.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "docs/paper/physica_scripta/figs"; PF = ROOT / "paper_figs/fig_masking_cut_compiled_v4/out"
PF.mkdir(parents=True, exist_ok=True)
OLD_PDF_W_PT = 2396.88                          # page width of figs/fig_masking_cut_compiled_final.pdf (same overall size)
TMP = Path(tempfile.mkdtemp())
COLOR = {"original": "#8d97a9", "orig2q": "#8d97a9", "twirl": "#E08A2E", "decoy_init": "#6A9C89",
         "decoy_h": "#6A9C89", "cut_rx": "#B8001F", "dd_pad": "#c6767f", "padlock": "#384B70"}   # paper_figs/qmaskcut_style.py
WHITE_TXT = {"padlock", "cut_rx"}
LEGEND = [("original", "original circuit gate"), ("twirl", "QOTP / Pauli twirl"),
          ("decoy_init", "decoy initialisation / final H (native gates)"),
          ("dd_pad", "DD masking pulse (rz, x, rz)"), ("padlock", "entangling padlock (CZ)"),
          ("cut_rx", "gates holding the cut RX(±t), merged by the compiler")]
NEUTRAL, DECOY_C, OUT_C, CUTC = "#eef0f4", "#6A9C89", "#FFF5E4", "#B8001F"

# ---- the run's circuits and the device target in force when they were compiled
if CACHE.exists():
    P = pickle.load(open(CACHE, "rb"))
else:
    if RUN_PKL.exists():
        R = pickle.load(open(RUN_PKL, "rb"))
    else:                                       # public release: the same groups and circuits, without prepared.pkl
        import gzip
        from qiskit import qpy
        with gzip.open(RUN_PKL.parent / "submitted_circuits.qpy.gz", "rb") as fh:
            R = {"groups": pickle.load(open(RUN_PKL.parent / "groups.pkl", "rb")), "circuits": qpy.load(fh)}
    G0 = next(g for g in R["groups"] if g["circuit"] == NAME)
    P = {"group": G0, "circuits": {i: R["circuits"][i] for i in [G0["orig"]] + G0["frag"]}}
    del R; pickle.dump(P, open(CACHE, "wb"))
G, SUB = P["group"], P["circuits"]
from qiskit_ibm_runtime import QiskitRuntimeService
acct = json.loads((RUN_PKL.parent / "jobs.json").read_text())["account"]
assert acct != "pknu3"
bk = QiskitRuntimeService(name=acct).backend("ibm_marrakesh")            # read only
t = bk.target_history(datetime=PREP_TIME); CAL = str(bk.properties(datetime=PREP_TIME).last_update_date)
gmap = good_map(t); region = len({q for e in gmap.get_edges() for q in e})

path_q = next(p for nm, p, _ in V4.CIRCS if nm == NAME); pieces = next(k for nm, _, k in V4.CIRCS if nm == NAME)
base = load(NAME, path_q); n = base.num_qubits

# ---- rebuild the protected circuit exactly as hw_final_v4.py did, and check it against the stored one
ft, best_keys = {}, None
if n <= 14:
    idl = Statevector(base).probabilities()
    for f in itertools.product([0, 1], repeat=n):
        if any(f):
            kk = np.arange(2 ** n) ^ sum(b_ << q for q, b_ in enumerate(f)); ft["".join(map(str, f))] = 0.5 * np.abs(idl - idl[kk]).sum()
    best_keys = {x for x, y in ft.items() if y > max(ft.values()) - 1e-9}
pick = None
for sd in range(cfg.SEED, cfg.SEED + (300 if best_keys else 1)):
    prot, tags, info = protect_general(base, n_decoy=2, n_padlock=V4.PADLOCKS, reps=2, seed=sd, n_twirl="random")
    v = ft.get(info["flip"], 0.0) if best_keys else 0.0
    if pick is None or v > pick[0] + 1e-12:
        pick = (v, sd, prot, tags, info)
    if not best_keys or info["flip"] in best_keys:
        break
key_tvd, key_seed, prot, tags, info = pick
lab = QuantumCircuit(*prot.qregs)
for ins, tg in zip(prot.data, tags):
    if tg != "dd_pad":
        lab.append(ins.operation, ins.qubits)
for d in info["decoy"]:
    lab.append(HGate(), [d])
N = lab.num_qubits
CHECKS = []
def check(ok, what):
    CHECKS.append((bool(ok), what)); print(("OK       " if ok else "MISMATCH ") + what, flush=True)
check(key_seed == G["key_seed"] and info["flip"] == G["key"] == G["info"]["flip"], f"key {info['flip']} seed {key_seed} (stored {G['key']} seed {G['key_seed']})")
check(lab.size() == G["lab"].size() and lab == G["lab"], f"protected circuit {lab.size()} gates (stored {G['lab'].size()}), gate by gate")
check(info["decoy"] == G["info"]["decoy"] and info["real"] == G["info"]["real"], f"decoy wires {info['decoy']}, data wires {info['real']}")
cmap = gmap if N <= region else W.working_map(t)

# original: recompiled on the run-time target must equal the submitted one
b0 = base.copy(); b0.measure_all(); c_orig_re = W.best(b0, t, cmap)
c_orig = SUB[G["orig"]]
check(c_orig_re == c_orig, f"original recompiled on calibration {CAL} equals the submitted circuit")

# fragments: cut + RX pair + compile + twirl + DD, as the run did, must equal the submitted variants
if pieces:
    cq_re, _ = C.find_balanced_cut(lab, pieces, max_cuts=2, allow_extra_cut=False)
else:
    cq_re, _ = C.cut_to_width(lab, 20 if n < 40 else 27)
rr = random.Random(1000 + key_seed); subs = [sc.copy() for sc in cq_re.cut_solution["subcircuits"]]
pre = {i_: [] for i_ in range(len(subs))}; rx_t = []
for w_, pth in C.cpm_index(cq_re).items():
    for (su, qu), (sdd, qd) in zip(pth, pth[1:]):
        t_ = rr.uniform(0, 2 * math.pi); subs[su].append(RXGate(t_), [qu]); pre[sdd].append((qd, -t_)); rx_t.append((w_, su, sdd, t_))
for s_, items in pre.items():
    if items:
        h_ = QuantumCircuit(*subs[s_].qregs)
        for qd, t_ in items:
            h_.append(RXGate(t_), [qd])
        subs[s_] = h_.compose(subs[s_])
cq_re.cut_solution["subcircuits"] = subs
labels_re, frags = C.instance_circuits(cq_re)
frags = [f if f.num_clbits else f.measure_all(inplace=False) for f in frags]
check(labels_re == G["labels"] and all(a == b for a, b in zip(subs, G["subs"])), f"cut: {len(subs)} fragments, {len(labels_re)} variants, RX pair angles reproduced")
comp = [None] * len(frags)
for s in sorted({x[0] for x in labels_re}):
    idx = [j for j, x in enumerate(labels_re) if x[0] == s]
    rep = W.best(frags[idx[0]], t, cmap)
    lay = rep.layout.initial_index_layout(filter_ancillas=True)[:frags[idx[0]].num_qubits]
    for j in idx:
        comp[j] = W.tp(frags[j], t, cmap, W.SEED, layout=lay)
masked_re = [W.mask_fill(W.twirl_smart(c, t, cfg.SEED + j), t, cfg.SEED + j, max_pairs=V4.MAX_PAIRS) for j, c in enumerate(comp)]
masked = [SUB[i] for i in G["frag"]]
check(all(a == b for a, b in zip(masked_re, masked)), "every fragment variant rebuilt equals the submitted one")

# (b): the whole protected circuit (never submitted uncut), compiled on the same target, twirl + DD as the run
p = lab.copy(); p.measure_all(); cw = W.best(p, t, cmap)
whole = W.mask_fill(W.twirl_smart(cw, t, cfg.SEED), t, cfg.SEED, max_pairs=V4.MAX_PAIRS)
go = W.gates_only(c_orig); n3 = ng(go, W.gates_only(whole))
cq = C._finish(G["lab"], G["subs"], G["cpm"]); K = G["K"]; labels = G["labels"]


def meas_map(c):
    """Physical qubit of each measured bit (clbit k = local qubit k of the circuit before compilation)."""
    m = {c.find_bit(i.clbits[0]).index: c.find_bit(i.qubits[0]).index for i in c.data if i.operation.name == "measure"}
    return [m[k] for k in range(len(m))]


def role(v):                                  # v: a wire of the protected circuit
    if v in info["decoy"]:
        return f"decoy {info['decoy'].index(v) + 1}"
    d = info["real"].index(v)
    return "ancilla" if d == n - 1 else f"data {d}"


def gate_list(tc, decoy_phys=(), cut_in=(), cut_out=()):
    """[(label, [physical qubits], colour tag, k)] with masking runs merged per wire (EXPAND=0).
    Decoy H: the native single-qubit gates after the last two-qubit gate on a decoy wire. Cut RX: the native
    single-qubit gates before the first two-qubit gate on an incoming cut wire / after the last one on an
    outgoing cut wire (the compiler merges RX(+-t) with its neighbours, so this run carries it)."""
    ops = [i for i in tc.data if i.operation.name not in ("barrier", "measure", "delay")]
    q2 = {}
    for k, i in enumerate(ops):
        if len(i.qubits) == 2:
            for q in i.qubits:
                q2.setdefault(tc.find_bit(q).index, []).append(k)
    out, last, seen = [], {}, set()
    for k, i in enumerate(ops):
        nm, lb = i.operation.name, getattr(i.operation, "label", None)
        if lb == "mask" and nm == "rz" and not REAL:
            continue
        qs = [tc.find_bit(q).index for q in i.qubits]
        if lb == "mask":
            j = last.get(qs[0])
            if not EXPAND and j is not None and out[j][2] == "dd_pad":
                out[j][3] += 1; continue
            out.append([nm if REAL else "DD", qs, "dd_pad", 1]); last[qs[0]] = len(out) - 1; continue
        elif lb == "twirl":
            out.append(["Z" if nm == "rz" else nm, qs, "twirl", 1])
        elif len(qs) == 2 and set(qs) & set(decoy_phys):
            out.append([nm, qs, "padlock", 1])
        elif nm == "x" and qs[0] in decoy_phys and qs[0] not in seen:
            out.append([nm, qs, "decoy_init", 1])
        elif len(qs) == 1 and qs[0] in decoy_phys and q2.get(qs[0]) and k > q2[qs[0]][-1]:
            out.append([nm, qs, "decoy_h", 1])
        elif len(qs) == 1 and qs[0] in cut_out and (not q2.get(qs[0]) or k > q2[qs[0]][-1]):
            out.append([nm, qs, "cut_rx", 1])
        elif len(qs) == 1 and qs[0] in cut_in and (not q2.get(qs[0]) or k < q2[qs[0]][0]):
            out.append([nm, qs, "cut_rx", 1])
        else:
            out.append([nm, qs, "orig2q" if len(qs) == 2 else "original", 1])
        for q in qs:
            last[q] = len(out) - 1; seen.add(q)
    return out


def asap(gates, wires):
    free, out = {w: 0 for w in wires}, []
    for _, qs, _, _ in gates:
        c = max(free[w] for w in qs)
        for w in qs:
            free[w] = c + 1
        out.append(c)
    return out


def panel(gates, wires, names, title, stem, T, cut_in=(), cut_out=(), decoys=()):
    mom = asap(gates, wires)
    L = ['digraph G {', 'rankdir=LR; ranksep=0.18; nodesep=0.12; newrank=true; splines=true;',
         'node [shape=ellipse,style=filled,fontname="Liberation Serif",fontsize=17,'
         'width=0.72,height=0.46,fixedsize=true,penwidth=0.9];',
         'edge [arrowsize=0.55,color="#6b6b6b",penwidth=0.9];',
         f'labelloc="t"; labeljust="l"; label="{title}"; fontname="Liberation Serif Bold"; fontsize=24;']
    last = {}
    for w in wires:
        if w in cut_in:
            L.append(f'in{w} [label="{names[w].split(chr(10))[0]}",shape=octagon,fillcolor="white",color="{CUTC}",'
                     f'penwidth=2.5,fontcolor="{CUTC}",width=1.05,height=0.56,group=w{w}];')
        else:
            d = w in decoys
            L.append(f'in{w} [label="{names[w].replace(chr(10), chr(92) + "n")}",shape=box,style="filled,rounded",'
                     f'fillcolor="{DECOY_C if d else NEUTRAL}",color="{"#3f6b5b" if d else "#8d97a9"}",'
                     f'penwidth={3 if d else 1},width=1.05,height=0.56,group=w{w}];')
        last[w] = f"in{w}"
    cols = {c: [] for c in range(T)}
    for i, ((nm, qs, tg, k), c) in enumerate(zip(gates, mom)):
        nid = f"g{i}"; cols[c].append((wires.index(min(qs, key=wires.index)), nid))
        txt = (f"DD×{k}" if k > 1 else nm) if tg == "dd_pad" else nm
        wd = ',width=1.15,fontsize=15' if tg == "dd_pad" and not REAL else ''
        L.append(f'{nid} [label="{txt}",fillcolor="{COLOR[tg]}"{wd},' + ('fontcolor="white",' if tg in WHITE_TXT else '') + f'group=w{min(qs, key=wires.index)}];')
        for w in qs:
            L.append(f'{last[w]} -> {nid};'); last[w] = nid
    for c in range(T):
        L.append(f'r{c} [style=invis,label="",width=0.56,height=0.02];')
    L.append('rin [style=invis,label="",width=0.95,height=0.02];')
    L.append('rout [style=invis,label="",width=0.95,height=0.02];')
    L.append(" -> ".join(["rin"] + [f"r{c}" for c in range(T)] + ["rout"]) + " [style=invis];")
    for w in wires:
        top = names[w].split("\n")[0]
        if w in cut_out:
            L.append(f'out{w} [label="{top}",shape=octagon,fillcolor="white",color="{CUTC}",'
                     f'penwidth=2.5,fontcolor="{CUTC}",width=1.05,height=0.56,group=w{w}];')
        else:
            L.append(f'out{w} [label="{top}",shape=box,style="filled,rounded",fillcolor="{OUT_C}",'
                     f'color="#7a0015",width=1.05,height=0.56,group=w{w}];')
        L.append(f'{last[w]} -> out{w};')
    row = {w: [f"in{w}"] for w in wires}
    for i, (_, qs, _, _) in enumerate(gates):
        row[min(qs, key=wires.index)].append(f"g{i}")
    for w in wires:
        L.append(" -> ".join(row[w] + [f"out{w}"]) + " [style=invis,weight=60];")
    rank = lambda ids: "{rank=same; " + " ".join(ids) + "; " + " -> ".join(ids) + " [style=invis];}"
    L.append(rank(["rin"] + [f"in{w}" for w in wires]))
    L.append(rank(["rout"] + [f"out{w}" for w in wires]))
    for c in range(T):
        L.append(rank([f"r{c}"] + [nid for _, nid in sorted(cols[c])]) if cols[c] else "{rank=same; r%d;}" % c)
    L.append("}")
    (TMP / f"{stem}.dot").write_text("\n".join(L))
    subprocess.run(["dot", "-Tpng", "-Gdpi=200", str(TMP / f"{stem}.dot"), "-o", str(TMP / f"{stem}.png")], check=True)
    return TMP / f"{stem}.png"


def us(c):
    return round(c.estimate_duration(t, unit="u"), 3)


def counts(c):
    lb = lambda i: getattr(i.operation, "label", None)
    return {"mask_pulses": sum(1 for i in c.data if i.operation.name == "x" and lb(i) == "mask"),
            "mask_gates": sum(1 for i in c.data if lb(i) == "mask"),
            "twirl_gates": sum(1 for i in c.data if lb(i) == "twirl"),
            "twirl_x": sum(1 for i in c.data if lb(i) == "twirl" and i.operation.name == "x"),
            "twirl_rz": sum(1 for i in c.data if lb(i) == "twirl" and i.operation.name == "rz")}


# ---- wires and their names: physical qubit, role
po = meas_map(c_orig)[:n]
pw = cw.layout.initial_index_layout(filter_ancillas=True)[:N]
dec_w = {pw[v] for v in info["decoy"]}
ga = gate_list(c_orig)
gb = gate_list(whole, dec_w)
path = C.cpm_index(cq)                       # wire -> [(fragment, local qubit), ...]
fr, REPORT = [], {"circuit": NAME, "calibration": CAL, "target": f"target_history({PREP_TIME.isoformat()}), account {acct}",
                  "key": info["flip"], "key_seed": key_seed, "key_tvd": round(key_tvd, 4), "K": K,
                  "a_original": {"native_gates": go.size(), "layers": go.depth(), "duration_us": us(c_orig),
                                 "ops": dict(go.count_ops())},
                  "b_protected_logical": {"gates": G["lab"].size(), "layers": G["lab"].depth(), "ops": dict(G["lab"].count_ops())},
                  "b_compiled_masked": {"native_gates": W.gates_only(whole).size(), "layers": W.gates_only(whole).depth(),
                                        "duration_us": us(whole), "normGED_vs_a": round(n3, 4),
                                        "native_gates_before_masking": W.gates_only(cw).size(),
                                        "layers_before_masking": W.gates_only(cw).depth(), **counts(whole)},
                  "cuts": [], "fragments": []}
for w, segs in path.items():
    for (s1, _), (s2, _) in zip(segs, segs[1:]):
        REPORT["cuts"].append({"wire": w, "role": role(w), "ends": f"F{s1 + 1}", "starts": f"F{s2 + 1}"})
for s in sorted({x[0] for x in labels}):
    js = [j for j, x in enumerate(labels) if x[0] == s]
    j = max(js, key=lambda j: masked[j].estimate_duration(t))
    lay = meas_map(masked[j])
    loc = {l: w for w, segs in path.items() for s2, l in segs if s2 == s}       # local -> protected wire
    phys = {w: lay[l] for l, w in loc.items()}
    cin = {phys[w] for w, segs in path.items() for k, (s2, _) in enumerate(segs) if s2 == s and k > 0}
    cout = {phys[w] for w, segs in path.items() for k, (s2, _) in enumerate(segs) if s2 == s and k < len(segs) - 1}
    dph = {phys[v] for v in info["decoy"] if v in phys}
    g = gate_list(masked[j], dph, cin, cout)
    fr.append((s, j, phys, cin, cout, dph, g))
    c = masked[j]
    REPORT["fragments"].append({
        "fragment": f"F{s + 1}", "wires": {role(w): phys[w] for w in sorted(phys)}, "variants": len(js),
        "drawn_variant": [str(x) for x in labels[j][1:]], "native_gates": W.gates_only(c).size(),
        "native_gates_without_masking": W.gates_only(W.strip_mask(c)).size(),
        "layers": W.gates_only(c).depth(), "duration_us": us(c), "padlocks": sum(1 for x in g if x[2] == "padlock"),
        "decoys": [role(v) for v in info["decoy"] if v in phys], **counts(c),
        "per_variant": [{"variant": [str(x) for x in labels[jj][1:]], "native_gates": W.gates_only(masked[jj]).size(),
                         "layers": W.gates_only(masked[jj]).depth(), "duration_us": us(masked[jj]), **counts(masked[jj])} for jj in js]})
T = max(max(asap(g, sorted({q for _, qs, _, _ in g for q in qs}))) + 1 for g in [ga, gb] + [f[6] for f in fr])
panels = [panel(ga, [po[v] for v in range(n)], {po[v]: ('ancilla' if v == n - 1 else f'data {v}') for v in range(n)},
                f"(a)  ORIGINAL {NAME}, compiled (optimization_level=3), as submitted   -   {go.size()} native gates, "
                f"depth {go.depth()}, {us(c_orig):.2f} us", "a", T),
          panel(gb, [pw[v] for v in range(N)], {pw[v]: role(v) for v in range(N)},
                f"(b)  Q-MaskCut PROTECTED, before cutting, compiled + twirl + no-pattern DD   -   "
                f"{W.gates_only(whole).size()} native gates, depth {W.gates_only(whole).depth()}, {us(whole):.2f} us, normGED {n3:.3f}",
                "b", T, decoys=dec_w)]
for s, j, phys, cin, cout, dph, g in fr:
    wires = [phys[w] for w in sorted(phys)]
    c = masked[j]
    panels.append(panel(g, wires, {phys[w]: role(w) for w in phys},
                        f"({chr(99 + s)})  FRAGMENT F{s + 1}, longest variant, as submitted (compiled + twirl + no-pattern DD)   -   "
                        f"{W.gates_only(c).size()} native gates, depth {W.gates_only(c).depth()}, {us(c):.2f} us",
                        f"f{s}", T, cut_in=cin, cut_out=cout, decoys=dph))

# ---- legend and assembly
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"]})
from matplotlib.lines import Line2D
h = [Line2D([], [], ls="", marker="o", markersize=15, markerfacecolor=COLOR[k], markeredgecolor="#222222",
            markeredgewidth=0.6, label=v) for k, v in LEGEND]
h += [Line2D([], [], ls="", marker="8", markersize=16, markerfacecolor="white", markeredgecolor=CUTC,
             markeredgewidth=2, label=f"cut wire (K = {K})"),
      Line2D([], [], ls="", marker="s", markersize=15, markerfacecolor=DECOY_C, markeredgecolor="#3f6b5b",
             markeredgewidth=1.5, label="decoy wire")]
lf = plt.figure(figsize=(26, 1.2))
lf.legend(handles=h, loc="center", ncol=4, frameon=False, fontsize=17, handletextpad=0.25, columnspacing=1.4)
lf.savefig(TMP / "leg.png", dpi=200, bbox_inches="tight", facecolor="white"); plt.close(lf)


def tight(im, pad=24):
    a = np.asarray(im.convert("L")); ys, xs = np.where(a < 250)
    c = im.crop((max(0, xs.min() - 3), max(0, ys.min() - 3), min(im.width, xs.max() + 4), min(im.height, ys.max() + 4)))
    o = Image.new("RGB", (c.width + 2 * pad, c.height + 2 * pad), "white"); o.paste(c, (pad, pad))
    return o


ims = [tight(Image.open(p).convert("RGB")) for p in panels]
for k, im in enumerate(ims):
    im.save(OUT / f"dag_{NAME}_v4_panel_{chr(97 + k)}{'' if EXPAND else '_compact'}.png")
lg = tight(Image.open(TMP / "leg.png").convert("RGB"), pad=8)
Wd = max(max(i.width for i in ims), lg.width)
out = Image.new("RGB", (Wd, sum(i.height for i in ims) + lg.height + 30), "white"); y = 0
for i in ims:
    out.paste(i, (0, y)); y += i.height
out.paste(lg, ((Wd - lg.width) // 2, y + 10))
out.save(OUT / f"dag_{NAME}_paper_style_v4{'' if EXPAND else '_compact'}.png")
if EXPAND:                                     # the paper figure is the expanded, real-gate one
    res = out.width / OLD_PDF_W_PT * 72        # PDF page as wide as the old figure's
    for d in [p for p in (FIG, PF) if p.is_dir()]:   # the paper's folder is not in the public release
        out.save(d / "fig_masking_cut_compiled_v4.png"); out.save(d / "fig_masking_cut_compiled_v4.pdf", resolution=res)
    (PF / "numbers.json").write_text(json.dumps(REPORT, indent=1, default=str))
    (PF / "checks.txt").write_text("\n".join(("OK       " if ok else "MISMATCH ") + w for ok, w in CHECKS) + "\n")
print(json.dumps(REPORT, indent=1, default=str))
print(f"figure {out.size[0]}x{out.size[1]} px; checks: {sum(ok for ok, _ in CHECKS)}/{len(CHECKS)} OK")
if not all(ok for ok, _ in CHECKS):
    print("SOME CHECKS FAILED: see above"); sys.exit(2)
