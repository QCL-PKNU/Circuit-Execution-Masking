#!/usr/bin/env python3
"""protect_v2: single-CZ padlocks with the correction folded into the classical key.

Change from the current design
------------------------------
The old padlock is CZ . T-X-T-X . CZ, which is identity, so no correction is needed -
but identity is exactly what an optimiser deletes: measured 0% survival at opt-3, and it
costs +22 depth on BV-7.

v2 uses a SINGLE CZ per padlock. That is not identity, so it cannot be optimised away
(measured 100% survival). To keep the result recoverable we hold the decoy in a
computational basis state at padlock time, where CZ(decoy,target) acts as Z(target) iff
the decoy bit is 1. For BV the final H turns that Z into a bit flip, so the correction is
one more XOR on top of the twirl key - no extra quantum work.

Decoy masking therefore uses X and Z only, which keep the decoy in the computational
basis; data wires keep the full T/SX/Y masking set.
"""
from qiskit import QuantumCircuit
from qiskit.circuit.library import XGate, TGate, SXGate, ZGate, YGate, CZGate

def protect_v2(qc, n_decoy=2, n_padlock=2, reps=1, seed=42):
    """returns (protected_circuit, tags, info) where info['flip'] is the extra XOR mask."""
    import random as _r
    rng=_r.Random(seed)
    n=qc.num_qubits; N=n+n_decoy
    n_data=n-1                                        # last qubit is the BV ancilla
    real=sorted(rng.sample(range(N), n))              # interleaved, never contiguous
    DEC=sorted(set(range(N))-set(real))
    pos={i:real[i] for i in range(n)}
    # twirl a DATA qubit, never the ancilla: an X on the ancilla cancels the circuit's
    # own X and destroys the phase kickback entirely
    twirl_q=rng.randrange(n_data)
    out=QuantumCircuit(N); tags=[]
    def app(op,qs,t): out.append(op,[out.qubits[q] for q in qs]); tags.append(t)

    flip=[0]*n_data
    app(XGate(),[pos[twirl_q]],"twirl")   # QOTP in; the matching X at the end cancels it
    dbit={}
    for i,d in enumerate(DEC):
        app(XGate(),[d],"decoy_init"); dbit[d]=1            # |1> so CZ acts as Z

    data_wires=[pos[q] for q in range(n_data)]
    npl=min(n_padlock, len(data_wires))
    placed=False

    def place_padlocks():
        """CZ(decoy|1>, data) acts as Z. Measured: a no-op before the H layer, and a clean
        bit flip once the data qubit is in superposition - so it must go AFTER the H."""
        nonlocal placed
        if placed: return
        placed=True
        for k in range(npl):
            d=DEC[k % len(DEC)]; t=data_wires[k]
            app(CZGate(),[d,t],"padlock")
            if dbit[d]==1:
                flip[[q for q,w in pos.items() if w==t][0]] ^= 1

    for inst in qc.data:
        if inst.operation.num_qubits==2: place_padlocks()   # first 2q gate = H layer done
        app(inst.operation,[pos[qc.find_bit(q).index] for q in inst.qubits],"original")
        if inst.operation.num_qubits==2:
            busy={pos[qc.find_bit(q).index] for q in inst.qubits}
            for k,q in enumerate([x for x in range(N) if x not in busy]):
                for _ in range(reps):
                    if q in DEC:
                        app(XGate(),[q],"dd_pad"); app(ZGate(),[q],"dd_pad"); dbit[q]^=1
                    elif k%3==0: app(TGate(),[q],"dd_pad"); app(XGate(),[q],"dd_pad")
                    elif k%3==1: app(SXGate(),[q],"dd_pad"); app(ZGate(),[q],"dd_pad")
                    else: app(XGate(),[q],"dd_pad"); app(YGate(),[q],"dd_pad")
    place_padlocks()
    app(XGate(),[pos[twirl_q]],"twirl")
    return out, tags, {"real":real,"decoy":DEC,"twirl_q":twirl_q,
                       "flip":"".join(map(str,flip)),"n_padlock":npl}
