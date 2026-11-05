"""normGED and GED_2Q: the paper's Eq. 14, with GED by networkx search or by exact-cost mapping bounds."""
from __future__ import annotations

import random
import time

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit

from .. import config as cfg
from ..utils.circuits import two_q_only, unmeasured
from ..utils.logs import log

def circuit_to_minimal_dag(qc: QuantumCircuit) -> nx.DiGraph:
    """Gate-dependency DAG: one node per gate, an edge to the NEXT gate on each of its qubits."""
    G = nx.DiGraph()
    last: dict[int, int] = {}
    for idx, inst in enumerate(qc.data):
        if inst.operation.name == "barrier":
            continue
        G.add_node(idx, op=inst.operation.name)
        for q in (qc.find_bit(b).index for b in inst.qubits):
            if q in last:
                G.add_edge(last[q], idx)
            last[q] = idx
    return G


def normged_paper_eq14(qc_orig, qc_prot, timeout=20):
    """Eq. 14 of the ECQCO paper, verbatim: GED / (max(|V|,|V'|) + max(|E|,|E'|)).

    GED by networkx; if it cannot finish within `timeout`, its first upper bound is used.
    Returns (normGED, seconds, denominator).
    """
    Ga, Gb = circuit_to_minimal_dag(qc_orig), circuit_to_minimal_dag(qc_prot)
    denom = (max(Ga.number_of_nodes(), Gb.number_of_nodes())
             + max(Ga.number_of_edges(), Gb.number_of_edges()))
    if denom == 0:
        return 0.0, 0.0, denom
    t0 = time.time()
    ged = nx.graph_edit_distance(Ga, Gb, timeout=timeout)
    if ged is None:
        ged = next(iter(nx.optimize_graph_edit_distance(Ga, Gb)))
    return float(ged) / denom, time.time() - t0, denom


def _edit_cost(A, B, m: dict) -> int:
    """Exact edit cost of the node mapping m (A -> B); unmapped nodes are deleted/inserted."""
    kept = sum(1 for u, v in A.edges if u in m and v in m and B.has_edge(m[u], m[v]))
    return ((A.number_of_nodes() - len(m)) + (B.number_of_nodes() - len(m))
            + (A.number_of_edges() - kept) + (B.number_of_edges() - kept))


def _tokens(qc: QuantumCircuit, G, by: str) -> list:
    """Per DAG node (= instruction index), its gate name or the tuple of qubits it acts on."""
    if by == "name":
        return [G.nodes[v]["op"] for v in sorted(G.nodes)]
    return [tuple(qc.find_bit(q).index for q in qc.data[v].qubits) for v in sorted(G.nodes)]


def _lcs_map(A, B, la: list, lb: list) -> dict:
    """Order-preserving map: longest common subsequence of the node tokens in circuit order.

    Obfuscation mostly INSERTS gates around the original ones, so matching the original
    sequence inside the obfuscated one keeps most dependency edges.
    """
    a = sorted(A.nodes); b = sorted(B.nodes)
    n1, n2 = len(a), len(b)
    prev = [0] * (n2 + 1)
    back = []
    for i in range(1, n1 + 1):
        cur = [0] * (n2 + 1)
        row = bytearray(n2 + 1)
        x = la[i - 1]
        for j in range(1, n2 + 1):
            if x == lb[j - 1]:
                cur[j] = prev[j - 1] + 1; row[j] = 1
            elif prev[j] >= cur[j - 1]:
                cur[j] = prev[j]; row[j] = 2
            else:
                cur[j] = cur[j - 1]; row[j] = 3
        back.append(row); prev = cur
    m, i, j = {}, n1, n2
    while i and j:
        d = back[i - 1][j]
        if d == 1:
            m[a[i - 1]] = b[j - 1]; i -= 1; j -= 1
        elif d == 2:
            i -= 1
        else:
            j -= 1
    return m


def _degree_map(A, B) -> dict:
    """Bipartite assignment on a degree-aware cost matrix (Riesen & Bunke 2009)."""
    from scipy.optimize import linear_sum_assignment
    a, b = list(A.nodes), list(B.nodes)
    n1, n2 = len(a), len(b)
    din_a = np.array([A.in_degree(v) for v in a]); dout_a = np.array([A.out_degree(v) for v in a])
    din_b = np.array([B.in_degree(v) for v in b]); dout_b = np.array([B.out_degree(v) for v in b])
    cost = np.full((n1 + n2, n1 + n2), 1e9)
    cost[:n1, :n2] = 0.5 * (np.abs(din_a[:, None] - din_b[None, :])
                            + np.abs(dout_a[:, None] - dout_b[None, :]))
    cost[:n1, n2:][np.diag_indices(n1)] = 1 + 0.5 * (din_a + dout_a)
    cost[n1:, :n2][np.diag_indices(n2)] = 1 + 0.5 * (din_b + dout_b)
    cost[n1:, n2:] = 0.0
    rows, cols = linear_sum_assignment(cost)
    return {a[r]: b[c] for r, c in zip(rows, cols) if r < n1 and c < n2}


def bipartite_normged(qa: QuantumCircuit, qb: QuantumCircuit) -> float:
    """Eq. 14 with GED bounded from above by the best of two node mappings.

    networkx's exact search cannot take wide circuits (thousands of nodes; it hits the
    recursion limit). Same graphs and unit costs as normged_paper_eq14 (insert/delete 1,
    substitution free). Two mappings are tried, the bipartite degree assignment and an
    order-preserving gate-sequence match, and the edit cost of each is counted EXACTLY;
    the smaller is a true upper bound on GED, never an estimate.
    """
    A, B = circuit_to_minimal_dag(qa), circuit_to_minimal_dag(qb)
    maps = [_degree_map(A, B),
            _lcs_map(A, B, _tokens(qa, A, "name"), _tokens(qb, B, "name")),
            _lcs_map(A, B, _tokens(qa, A, "qubits"), _tokens(qb, B, "qubits"))]
    ged = min(_edit_cost(A, B, m) for m in maps)
    return ged / (max(A.number_of_nodes(), B.number_of_nodes())
                  + max(A.number_of_edges(), B.number_of_edges()))


DEGREE_MAP_MAX_NODES = 6000         # above this the assignment matrix exceeds ~300 MB


def _complete(A, B, m: dict) -> dict:
    """Map every remaining node of the smaller side. Substitution is free, so each new
    pair lowers the node cost by 2 and can never lower the number of kept edges."""
    m = dict(m)
    free_b = sorted(set(B.nodes) - set(m.values()))
    for u in sorted(set(A.nodes) - set(m)):
        if not free_b:
            break
        m[u] = free_b.pop(0)
    return m


def _refine(A, B, m: dict, sweeps: int, seed: int) -> dict:
    """Deterministic local search: move or swap node images while that keeps more edges.

    A candidate image for node u is any B-node adjacent to the image of one of u's
    neighbours, the only places a move can gain an edge. Seeded and sorted, so the same
    input always returns the same mapping.
    """
    rng = random.Random(seed)
    m = dict(m)
    inv = {b: a for a, b in m.items()}
    succ = {u: list(A.successors(u)) for u in A}
    pred = {u: list(A.predecessors(u)) for u in A}

    def kept(nodes):
        edges = {(u, x) for u in nodes for x in succ[u]} | {(x, u) for u in nodes for x in pred[u]}
        return sum(1 for p, q in edges if p in m and q in m and B.has_edge(m[p], m[q]))

    def move(u, b):                  # put u on b; whoever sat on b takes u's old image
        cur, w = m.get(u), inv.get(b)
        m[u], inv[b] = b, u
        if w is not None:
            if cur is None:
                del m[w]
            else:
                m[w], inv[cur] = cur, w
        elif cur is not None:
            del inv[cur]
        return cur, w

    def undo(u, b, cur, w):
        del inv[b]
        if w is not None:
            m[w], inv[b] = b, w
            if cur is not None:
                inv[cur] = u
        elif cur is not None:
            inv[cur] = u
        if cur is None:
            del m[u]
        else:
            m[u] = cur

    order = sorted(A.nodes)
    for _ in range(sweeps):
        rng.shuffle(order)
        improved = False
        for u in order:
            cand = set()
            for x in succ[u]:
                if x in m:
                    cand.update(B.predecessors(m[x]))
            for x in pred[u]:
                if x in m:
                    cand.update(B.successors(m[x]))
            cand.discard(m.get(u))
            best, gain = None, 0
            for b in sorted(cand):
                w = inv.get(b)
                aff = [u] if w is None else [u, w]
                before = kept(aff)
                cur, w2 = move(u, b)
                g = kept(aff) - before
                undo(u, b, cur, w2)
                if g > gain:
                    best, gain = b, g
            if best is not None:
                move(u, best)
                improved = True
        if not improved:
            break
    return m


def deterministic_normged(qa: QuantumCircuit, qb: QuantumCircuit) -> float:
    """Eq. 14 with a REPRODUCIBLE upper bound on GED.

    Starts from three mappings (bipartite degree assignment, order-preserving matches on
    gate names and on gate qubits), completes each, improves each by the seeded local
    search above, and counts the edit cost of each EXACTLY. The smallest is reported. Unlike
    networkx's time-limited search, the result does not depend on machine load: the same
    circuits give the same number on every run and every machine.
    """
    A, B = circuit_to_minimal_dag(qa), circuit_to_minimal_dag(qb)
    starts = [_lcs_map(A, B, _tokens(qa, A, "name"), _tokens(qb, B, "name")),
              _lcs_map(A, B, _tokens(qa, A, "qubits"), _tokens(qb, B, "qubits"))]
    if A.number_of_nodes() + B.number_of_nodes() <= DEGREE_MAP_MAX_NODES:
        starts.insert(0, _degree_map(A, B))    # an (n1+n2)^2 matrix: small graphs only
    ged = min(_edit_cost(A, B, _refine(A, B, _complete(A, B, m), cfg.GED_SWEEPS, cfg.SEED))
              for m in starts)
    return ged / (max(A.number_of_nodes(), B.number_of_nodes())
                  + max(A.number_of_edges(), B.number_of_edges()))


def structure(base: QuantumCircuit, obf: QuantumCircuit) -> dict:
    a, b = unmeasured(base), unmeasured(obf)
    t = time.time()
    if cfg.GED_METHOD == "networkx" and base.num_qubits <= cfg.GED_NX_UP_TO:
        # every number here is an upper bound on GED, so the smaller one is kept
        method = f"best of networkx search ({cfg.GED_TIMEOUT:g}s) and mapping bounds"
        ng = min(normged_paper_eq14(a, b, timeout=cfg.GED_TIMEOUT)[0], bipartite_normged(a, b))
        g2 = min(normged_paper_eq14(two_q_only(a), two_q_only(b), timeout=cfg.GED_TIMEOUT)[0],
                 bipartite_normged(two_q_only(a), two_q_only(b)))
    else:
        method = f"deterministic upper bound ({cfg.GED_SWEEPS} sweeps, seed {cfg.SEED})"
        ng = deterministic_normged(a, b)
        g2 = deterministic_normged(two_q_only(a), two_q_only(b))
    log.info(f"structure    : normGED {ng:.4f} | GED_2Q {g2:.4f}  "
             f"(paper Eq. 14, GED by {method}; {time.time() - t:.1f}s)")
    return {"normGED": ng, "GED_2Q": g2, "ged_method": method}
