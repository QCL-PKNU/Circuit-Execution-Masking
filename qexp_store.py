"""RunStore: save every circuit and every count file of an experiment once, reuse it, never overwrite.

    runs/<date>_<version>/
        config.json                      settings of the run (written once; a different config -> error)
        circuits/<backend>/<circuit>/<method>__<role>.qasm
        counts/<backend>/<circuit>/<method>__<role>.json   {"shots": int, "counts": {bitstring: n}, ...meta}
        metrics/<backend>/<circuit>.json one row per method (computed only from the saved files)

    store = RunStore("runs/tier1_tier2", config)
    if not store.has_counts(b, c, m, r): store.save_counts(b, c, m, r, counts, shots, **meta)
    counts = store.load_counts(b, c, m, r)

Bitstrings are qiskit's (last character = clbit 0). Writing an existing file raises unless it is identical.
"""
from __future__ import annotations
import json
from pathlib import Path


class RunStore:
    def __init__(self, root, config: dict | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        cfg = self.root / "config.json"
        if config is not None:
            if cfg.exists():
                old = json.loads(cfg.read_text())
                if old != json.loads(json.dumps(config)):
                    diff = {k: (old.get(k), config.get(k)) for k in set(old) | set(config) if old.get(k) != config.get(k)}
                    raise ValueError(f"{cfg} exists with different settings {diff}; use a new run folder")
            else:
                cfg.write_text(json.dumps(config, indent=1, sort_keys=True))
        self.config = json.loads(cfg.read_text()) if cfg.exists() else {}

    # ---- paths
    def _p(self, kind, backend, circuit, method, role, ext):
        safe = lambda s: str(s).replace(" ", "_").replace("/", "-")                    # noqa: E731
        return self.root / kind / safe(backend) / safe(circuit) / f"{safe(method)}__{safe(role)}.{ext}"

    def _write_once(self, path: Path, text: str):
        if path.exists():
            if path.read_text() != text:
                raise FileExistsError(f"{path} exists with different content (RunStore never overwrites)")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp"); tmp.write_text(text); tmp.replace(path)

    # ---- circuits
    def has_circuit(self, backend, circuit, method, role):
        return self._p("circuits", backend, circuit, method, role, "qasm").exists()

    def save_circuit(self, backend, circuit, method, role, qc):
        from qiskit import qasm2
        try:
            text = qasm2.dumps(qc)
        except Exception:                                    # noqa: BLE001 - e.g. delay with dt units
            from qiskit import qasm3
            text = "// OPENQASM 3\n" + qasm3.dumps(qc)
        self._write_once(self._p("circuits", backend, circuit, method, role, "qasm"), text)

    def load_circuit(self, backend, circuit, method, role):
        from qiskit import QuantumCircuit
        return QuantumCircuit.from_qasm_file(str(self._p("circuits", backend, circuit, method, role, "qasm")))

    # ---- counts
    def has_counts(self, backend, circuit, method, role):
        return self._p("counts", backend, circuit, method, role, "json").exists()

    def save_counts(self, backend, circuit, method, role, counts: dict, shots: int, **meta):
        rec = {"shots": int(shots), "counts": {str(k): int(v) for k, v in sorted(counts.items())}, **meta}
        if sum(rec["counts"].values()) != shots:
            raise ValueError(f"counts sum {sum(rec['counts'].values())} != shots {shots}")
        self._write_once(self._p("counts", backend, circuit, method, role, "json"), json.dumps(rec, indent=0, sort_keys=True))

    def load_counts(self, backend, circuit, method, role) -> dict:
        return json.loads(self._p("counts", backend, circuit, method, role, "json").read_text())

    def roles(self, backend, circuit, method=None):
        d = self.root / "counts" / str(backend) / str(circuit)
        out = sorted(p.stem for p in d.glob("*.json")) if d.exists() else []
        return [r for r in out if method is None or r.startswith(str(method).replace(" ", "_") + "__")]

    # ---- metrics (derived; may be recomputed, so these are versioned by overwrite-free names)
    def save_metrics(self, backend, circuit, rows: list[dict]):
        p = self.root / "metrics" / str(backend) / f"{circuit}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rows, indent=1, default=str))        # derived from saved counts: safe to rewrite

    def load_metrics(self, backend=None):
        rows = []
        for p in sorted((self.root / "metrics").glob(f"{backend or '*'}/*.json")):
            rows += json.loads(p.read_text())
        return rows
