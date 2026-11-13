"""
MLflow tracking (self-hosted: a local SQLite file, no server needed).

One MLflow run per pipeline run: params = flattened config, metrics = everything the steps return,
tags = data hash / embedding version / gate decision / promoted flag, artifacts = small JSON reports.

UI:  mlflow ui --backend-store-uri sqlite:///mlflow.db     -> http://127.0.0.1:5000
If mlflow is not installed or disabled in config, every call is a no-op (the pipeline still works).
"""
import json
import re


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = json.dumps(v) if isinstance(v, (list, tuple)) else v
    return out


class Tracker:
    def __init__(self, cfg, paths, run_id: str, log=print):
        self.on, self.log, self.mlflow = False, log, None
        mc = cfg.get("mlflow", {})
        if not mc.get("enabled", True):
            return
        try:
            import mlflow
        except ImportError:
            log("mlflow not installed -> tracking disabled (pip install mlflow)")
            return
        uri = mc["tracking_uri"]
        if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):   # make relative path repo-relative
            uri = "sqlite:///" + (paths.repo_root / uri[len("sqlite:///"):]).as_posix()
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(mc["experiment"])
        existing = mlflow.search_runs(filter_string=f"tags.run_id = '{run_id}'", output_format="list")
        self.run = mlflow.start_run(run_id=existing[0].info.run_id) if existing else mlflow.start_run(run_name=run_id)
        mlflow.set_tag("run_id", run_id)
        self.mlflow, self.on = mlflow, True
        log(f"mlflow: tracking to {uri} (experiment '{mc['experiment']}')")

    def _safe(self, fn, *args):
        """Tracking must never kill a training run: log the problem and carry on."""
        try:
            fn(*args)
        except Exception as e:                                   # noqa: BLE001
            self.log(f"mlflow warning ({fn.__name__}): {e}")

    def params(self, cfg):
        if self.on:
            flat = _flatten(cfg)
            already = self.mlflow.get_run(self.run.info.run_id).data.params
            self._safe(self.mlflow.log_params, {k: v for k, v in flat.items() if k not in already})

    def metrics(self, m: dict):
        if self.on and m:
            safe = {re.sub(r"[^\w\-./: ]", "_", k.replace("@", "_at_")): float(v)
                    for k, v in m.items() if isinstance(v, (int, float))}
            self._safe(self.mlflow.log_metrics, safe)

    def tags(self, t: dict):
        if self.on:
            self._safe(self.mlflow.set_tags, {k: str(v) for k, v in t.items()})

    def artifact(self, path):
        if self.on and path.exists():
            self._safe(self.mlflow.log_artifact, str(path))

    def end(self, status="FINISHED"):
        if self.on:
            self.mlflow.end_run(status=status)
