"""ECQCO with the verification-only simulation disabled -- a FAIR scaling test.

QCSO builds the obfuscated circuit at `qcso_for_baseline_circ` (line ~251) and
writes the QASM at line ~256. Between them it calls generate_baseline_circuit(),
which runs Simulator('mqvector', n) purely to report fidelity; its result feeds
only the return value, which callers here discard. Skipping it does NOT change
the obfuscated circuit -- it removes a 2^n memory allocation that is not part of
the obfuscation. The SVG dump is also skipped (it is a drawing, not output).

ENV: ecqco-env
"""
import os,sys,io,json,csv,random,argparse,warnings,contextlib,time
warnings.filterwarnings("ignore")
ECQ=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..","..","..",
                                  "external_benchmarks","ecqco"))   # ECQCO, unmodified
os.chdir(ECQ); sys.path.insert(0,ECQ); os.makedirs("result",exist_ok=True)
import QCSO as QCSOmod
from QCOO import QCOOManager
from scripts.utils import QCManager

# --- disable the two verification-only steps ---
class _Fake:                      # stands in for the counts object QCSO returns
    data={}
def _no_sim(circ_prep,qcso_manager,analog_frame,program_qubits,total_shots):
    return [_Fake()]
QCSOmod.generate_baseline_circuit=_no_sim
QCManager.save_circuit_svg=lambda self,*a,**k: None

ap=argparse.ArgumentParser()
ap.add_argument("--list",required=True); ap.add_argument("--out",required=True)
ap.add_argument("--seed",type=int,default=42)
a=ap.parse_args(); os.makedirs(a.out,exist_ok=True)
rows=[]
print(f"{'benchmark':<16}{'q':>5}{'QCOO':>9}{'QCSO(ADOA)':>13}{'total':>10}  status")
print("-"*66,flush=True)
for j in json.load(open(a.list)):
    nm,src,nq=j["name"],j["qasm"],j["qubits"]
    tag=os.path.join(a.out,nm)
    try:
        random.seed(a.seed)
        with contextlib.redirect_stdout(io.StringIO()):
            t0=time.perf_counter()
            q=QCOOManager(nq,src,tag); qp,_=q.QCOO(None)
            t1=time.perf_counter()
            QCSOmod.QCSO(qp,tag,lamb_param=True,total_shots=1)
            t2=time.perf_counter()
        rows.append(dict(benchmark=nm,qubits=nq,seconds_qcoo=round(t1-t0,4),
            seconds_qcso=round(t2-t1,4),seconds_total=round(t2-t0,4),status="ok"))
        print(f"{nm:<16}{nq:>5}{t1-t0:>8.3f}s{t2-t1:>12.3f}s{t2-t0:>9.3f}s  ok",flush=True)
    except Exception as e:
        rows.append(dict(benchmark=nm,qubits=nq,seconds_qcoo="",seconds_qcso="",
            seconds_total="",status=f"{type(e).__name__}: {str(e)[:70]}"))
        print(f"{nm:<16}{nq:>5}{'':>9}{'':>13}{'':>10}  FAIL {type(e).__name__}: {str(e)[:50]}",flush=True)
    with open(os.path.join(a.out,"timing_noverify.csv"),"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
ok=sum(1 for r in rows if r["status"]=="ok")
print("-"*66)
print(f"  {ok}/{len(rows)} completed with the fidelity simulation disabled")
