"""ObfusQate baselines: inverse, composite and delayed gates."""
from __future__ import annotations

import importlib.util
import random
import time


from .. import config as cfg
from ..utils.circuits import n2q
from ..device.backend import run_whole, semantics
from ..utils.logs import banner, describe, draw, log
from ..metrics.normged import structure

def _obf_module(name: str):
    spec = importlib.util.spec_from_file_location(f"obf_{name}", cfg.OBF_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


OBFUSQATE = {
    "inverse gates": lambda qc: _obf_module("InverseGates").obfuscate_circuit(qc, obfuscate=True),
    "composite gates": lambda qc: _obf_module("CompositeGates").insert_obfuscation(qc),
    "delayed gates": lambda qc: _obf_module("DelayedGates").insert_obfuscation(qc),
}


def run_obfusqate(dev, base, n, ref, technique) -> dict | None:
    banner(f"{base.name} | ObfusQate: {technique}", "-")
    random.seed(cfg.SEED)
    t = time.time()
    try:
        obf = OBFUSQATE[technique](base.copy())
    except Exception as e:      # noqa: BLE001
        log.info(f"FAILED       : {type(e).__name__}: {e}")
        return None
    log.info(f"compile      : {(time.time() - t) * 1e3:.1f} ms")
    describe(technique, obf)
    draw(obf, f"{technique} circuit")
    rec = {"scheme": technique, "qubits": obf.num_qubits, "gates": obf.size(),
           "depth": obf.depth(), "twoq": n2q(obf), "compile_s": time.time() - t}
    rec.update(structure(base, obf))
    rec["semantics"] = semantics(obf, n, ref, dev)
    rec.update(run_whole(dev, obf, n, ref))
    return rec
