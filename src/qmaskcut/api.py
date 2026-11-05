"""Public entry point: protect, cut, compile, mask, run, rebuild and decode one circuit, on simulators only.

protect_cut_run(circuit) runs the full Q-MaskCut method of the paper with the same functions as the paper's
experiments (qmaskcut_eval.py, Tier 1):

  1 key          X gates on data qubits; up to 14 qubits the best-TVD flip pattern (key seeds 42, 43, ... until
                 the flip is one of the patterns with the largest TVD to the exact output), random above
  2 protect      protect_general: 2 decoy wires, 2 CZ padlocks, then H on each decoy
  3 cut          CutQC into 2 balanced fragments (at most 2 cuts, one more if no balanced cut exists), with an
                 RX(+t) / RX(-t) pair across every cut wire (t uniform in [0, 2 pi))
  4 compile      every fragment variant at optimization_level=3, one layout per fragment (best of 8 by estimated
                 success probability) on the good-qubit coupling map
  5 mask         time-aware Pauli twirl, then no-pattern dynamical decoupling (at most 8 pulse pairs per idle
                 gap); every gap is checked to multiply to the identity
  6 run          noiseless (exact fragment probabilities of the masked circuits) and the backend's noise model
  7 rebuild      CutQC reconstruction from the fragment outputs
  8 decode       drop the decoys and XOR the key

No QPU job is submitted. The cut needs gurobipy (imported by CutQC); its size-limited licence is enough for small
circuits.
"""
from __future__ import annotations

import itertools, math, random, sys, time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SEED_STEP = 1_000_000          # Aer seeds shot s with seed + s, so simulator seeds are spaced this far apart
KEY_SEARCH = 300               # key seeds tried for the best-TVD key
BEST_KEY_MAX = 14              # qubits up to which the best-TVD key is searched


@dataclass
class Result:
    """What protect_cut_run returns. TVDs (paper Eq. 2) and the Hellinger fidelity are against the exact output
    of the original circuit."""
    name: str
    n: int                        # qubits of the original circuit
    key: str                      # one-time-pad key over the original qubits (flipped output bits)
    key_is_best: bool | None      # the key is a best-TVD flip pattern (None above BEST_KEY_MAX qubits)
    max_tvd: float | None         # largest TVD any flip pattern reaches for this circuit (the key's ceiling)
    cuts: int                     # CutQC cuts
    fragments: list[int]          # wires of each CutQC fragment (decoys included)
    variants: int                 # fragment variants (CutQC init/meas instances) compiled and run
    obfuscated: float             # TVD_obf: rebuilt output read without the key, noiseless (exact)
    restored: float               # TVD_deobf: decoys dropped and key removed, noiseless (exact): 0 up to rounding
    fidelity: float | None        # Hellinger fidelity of the decoded output on the noise model
    noisy_obfuscated: float | None = None
    noisy_restored: float | None = None
    backend: str = ""
    shots: int = 0
    seeds: list[int] = field(default_factory=list)
    seconds: float = 0.0
    circuits: list = field(default_factory=list, repr=False)   # the masked fragment variants as submitted
    protected: object = field(default=None, repr=False)          # protected circuit before cutting (key, decoys, padlocks, H)
    decoys: list = field(default_factory=list, repr=False)       # decoy wires of `protected`
    subcircuits: list = field(default_factory=list, repr=False)  # CutQC fragments with the RX pair, before compilation

    def summary(self) -> str:
        s = (f"{self.name}: {self.n} qubits, key {self.key}{' (best-TVD)' if self.key_is_best else ''}, "
             f"{self.cuts} cut(s), fragments {'+'.join(map(str, self.fragments))}, {self.variants} variants\n"
             f"  noiseless:   TVD_obf {self.obfuscated:.3f} (no key)   TVD_deobf {self.restored:.1e} (with key)")
        if self.max_tvd is not None and self.max_tvd < 0.99:
            s += f"   [highest TVD any key can reach here: {self.max_tvd:.3f}]"
        if self.fidelity is not None:
            s += (f"\n  {self.backend}: TVD_obf {self.noisy_obfuscated:.3f}   TVD_deobf {self.noisy_restored:.3f}   "
                  f"fidelity {self.fidelity:.3f}  ({self.shots} shots per variant)")
        return s


def load_qasmbench(name: str):
    """A benchmark circuit by name (e.g. "bell_n4"), prepared for cutting: QASMBench under <release>/QASMBench,
    or the MQT Bench circuits under <release>/data/circuits (bv_n3, bv_n4)."""
    from qmaskcut.utils.circuits import load
    own = ROOT / "data/circuits" / f"{name}.qasm"
    if own.exists():
        return load(name, str(own))
    hits = list((ROOT / "QASMBench").glob(f"*/{name}/{name}.qasm"))
    if not hits:
        raise FileNotFoundError(f"{name}.qasm not found under {ROOT / 'QASMBench'} or {ROOT / 'data/circuits'}")
    return load(name, str(hits[0].relative_to(ROOT / "QASMBench")))


def good_map(target, ro=0.05, q1=0.01, q2=0.05):
    """Coupling map without bad qubits/links (readout > ro, sx > q1, 2-qubit gate > q2), largest connected part."""
    import rustworkx as rx
    from qiskit.transpiler import CouplingMap
    import new_workflow_test as W
    g2 = W.two_q(target)
    ok = {q for q in range(target.num_qubits)
          if (target["measure"][(q,)].error or 0) <= ro and (target["sx"][(q,)].error or 0) <= q1}
    edges = [qs for qs, pr in target[g2].items() if qs[0] in ok and qs[1] in ok and pr is not None and (pr.error or 0) <= q2]
    keep = max(rx.connected_components(CouplingMap(edges).graph.to_undirected()), key=len)
    return CouplingMap([e for e in edges if e[0] in keep and e[1] in keep])


def exact_probs(sim, c):
    """Noiseless output of a compiled fragment variant as {bitstring: probability}, keyed like Qiskit counts
    (clbit 0 rightmost). Measurements are replaced by an exact save of the measured qubits."""
    from qiskit import QuantumCircuit
    meas = {}
    body = QuantumCircuit(*c.qregs)
    for ins in c.data:
        if ins.operation.name == "measure":
            meas[c.find_bit(ins.clbits[0]).index] = ins.qubits[0]
        elif ins.operation.name not in ("barrier", "delay"):
            body.append(ins.operation, ins.qubits)
    nc = c.num_clbits
    body.save_probabilities_dict([meas[k] for k in range(nc)], label="p")
    p = sim.run(body).result().data()["p"]
    return {format(k if isinstance(k, int) else int(k, 16), f"0{nc}b"): float(v) for k, v in p.items()}


def protect_cut_run(circuit, *, name: str = "circuit", backend=None, seed: int = 42, shots: int = 4000,
                    sim_seed: int = 1000, noise: bool = True, threads: int = 2) -> Result:
    """Run the full Q-MaskCut method on `circuit` (a QuantumCircuit without measurements, or one from
    load_qasmbench). backend: a Qiskit fake backend (default FakeMarrakesh). seed: first key seed (the paper
    uses 42). Noise-model variant j uses simulator seed sim_seed + j * 1,000,000. noise=False skips the
    noise-model run."""
    t0 = time.time()
    import numpy as np
    import new_workflow_test as W
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import HGate, RXGate
    from qiskit.quantum_info import Statevector
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime import fake_provider
    from qmaskcut.cutting import cutqc as C
    from qmaskcut.masking import protect_general

    W.BEST_OF = 8                                             # best of 8 compilations, as in the paper
    be = backend or fake_provider.FakeMarrakesh()
    t = be.target; cmap = good_map(t)
    base = C.strip_for_cutting(circuit); n = base.num_qubits
    ideal = Statevector(base).probabilities()                              # bit j = qubit j
    tvd = lambda d: float(0.5 * np.abs(np.asarray(d) - ideal).sum())        # noqa: E731
    fid = lambda d: float(np.sum(np.sqrt(np.clip(d, 0, None) * ideal)) ** 2)  # noqa: E731

    # 1 key: the flip patterns with the largest TVD to the exact output (up to BEST_KEY_MAX qubits)
    flip_tvd = {}
    if n <= BEST_KEY_MAX:
        for f in itertools.product([0, 1], repeat=n):
            if any(f):
                k = np.arange(2 ** n) ^ sum(b << j for j, b in enumerate(f))
                flip_tvd["".join(map(str, f))] = tvd(ideal[k])
    best = {k for k, v in flip_tvd.items() if v > max(flip_tvd.values()) - 1e-9} if flip_tvd else set()

    # 2 protect: key, decoys, padlocks (idle-wire padding left out: masking comes after compilation), H on decoys
    def protect(s):
        prot, tags, inf = protect_general(base, n_decoy=2, n_padlock=2, reps=2, seed=s, n_twirl="random")
        lab = QuantumCircuit(*prot.qregs)
        for ins, tg in zip(prot.data, tags):
            if tg != "dd_pad":
                lab.append(ins.operation, ins.qubits)
        for d in inf["decoy"]:
            lab.append(HGate(), [d])
        return lab, inf

    pick = None
    for s in range(seed, seed + (KEY_SEARCH if best else 1)):
        lab, info = protect(s)
        v = flip_tvd.get(info["flip"], 0.0)
        if pick is None or v > pick[0] + 1e-12:
            pick = (v, s, lab, info)
        if not best or info["flip"] in best:
            break
    _, kseed, lab, info = pick
    if len(C.independent_pieces(lab)) > 1:
        raise ValueError(f"{name}: the protected circuit falls into independent pieces; nothing to cut")

    # 3 cut, with an RX(+t) / RX(-t) pair across every cut wire
    cq, _ = C.find_balanced_cut(lab, 2, max_cuts=2, allow_extra_cut=True)
    r = random.Random(1000 + kseed); subs = [x.copy() for x in cq.cut_solution["subcircuits"]]
    pre = {i: [] for i in range(len(subs))}
    for w, path in C.cpm_index(cq).items():
        for (su, qu), (sd, qd) in zip(path, path[1:]):
            th = r.uniform(0, 2 * math.pi); subs[su].append(RXGate(th), [qu]); pre[sd].append((qd, -th))
    for s, items in pre.items():
        if items:
            h = QuantumCircuit(*subs[s].qregs)
            for qd, th in items:
                h.append(RXGate(th), [qd])
            subs[s] = h.compose(subs[s])
    cq.cut_solution["subcircuits"] = subs
    labels, circs = C.instance_circuits(cq)
    meas = [c.measure_all(inplace=False) for c in circs]

    # 4 compile (one layout per fragment) and 5 mask after compilation
    masked = [None] * len(meas)
    for s in sorted({x[0] for x in labels}):
        idx = [j for j, x in enumerate(labels) if x[0] == s]
        rep = W.best(meas[idx[0]], t, cmap)
        lay = rep.layout.initial_index_layout(filter_ancillas=True)[:circs[idx[0]].num_qubits]
        for j in idx:
            m = W.mask_fill(W.twirl_smart(W.tp(meas[j], t, cmap, W.SEED, layout=lay), t, kseed + j), t, kseed + j, max_pairs=8)
            W.strip_mask(m)                                   # every idle gap multiplies to the identity
            masked[j] = m

    # 7 rebuild and 8 decode
    def rebuild(outs):
        C.evaluate_from(cq, {l: C.counts_to_probs(o, c.num_qubits) for l, o, c in zip(labels, outs, circs)})
        dec = C.full_decoded(cq, info, n)[0]
        nok = C.full_decoded(cq, dict(info, flip="0" * n), n)[0]
        return dec, nok

    # 6 run: noiseless, exact probabilities of the masked circuits as submitted
    ideal_sim = AerSimulator(max_parallel_threads=threads)
    dec, nok = rebuild([exact_probs(ideal_sim, c) for c in masked])
    res = Result(name=name, n=n, key=info["flip"], key_is_best=(info["flip"] in best) if best else None,
                 max_tvd=max(flip_tvd.values()) if flip_tvd else None, cuts=cq.cut_solution["num_cuts"],
                 fragments=C.widths(cq), variants=len(masked), obfuscated=round(tvd(nok), 4), restored=tvd(dec),
                 fidelity=None, backend=be.name, circuits=masked,
                 protected=lab, decoys=list(info["decoy"]), subcircuits=subs)
    # 6 run: noise model, sampled
    if noise:
        sim = AerSimulator.from_backend(be, method="statevector", max_parallel_shots=1, max_parallel_threads=threads)
        seeds = [sim_seed + j * SEED_STEP for j in range(len(masked))]
        counts = [sim.run(c, shots=shots, seed_simulator=sd).result().get_counts() for c, sd in zip(masked, seeds)]
        dec, nok = rebuild(counts)
        res.fidelity, res.noisy_restored, res.noisy_obfuscated = round(fid(dec), 4), round(tvd(dec), 4), round(tvd(nok), 4)
        res.shots, res.seeds = shots, seeds
    res.seconds = round(time.time() - t0, 1)
    return res
