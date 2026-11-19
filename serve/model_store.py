"""
Where the engine gets its model from (P2 phase C). ModelManager asks `wanted()` every MODEL_WATCH_SECONDS and
`materialize()` only when the answer changed, in its background thread, while the old model keeps serving.

  LocalSource  models/<cat>/production.json on this disk -> runs/<id>/ + data/gold/<cat>/  (Phase 7d, unchanged)
  S3Source     the S3 pointer (pipeline/pointer.py) -> fetch run + gold snapshot into MODEL_CACHE_DIR,
               sha256-verified (pipeline/storage.py) -> load from there

MODEL_REGISTRY=local|s3 picks the source (the same switch as the pipeline). MODEL_NAMESPACE (optional) serves a
namespaced pointer, e.g. a demo, never the real one by accident.

What the S3 source refuses (the engine then keeps serving the old model):
  - a run without MANIFEST.json (unfinished upload) or with a file whose size/sha256 differs (storage.py)
  - a manifest that is not the one that was promoted: the pointer pins `manifest_sha256`
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from serve.config import ServeConfig


@dataclass(frozen=True)
class Target:
    """What production is, as seen by one poll. `key` identifies it; equal keys = nothing to do."""
    key: str
    run_id: str | None
    version: int | None = None
    entry: dict | None = None
    model_dir: Path | None = None        # known up front for local; filled by materialize() for S3


class ModelUnavailable(RuntimeError):
    """The pointer names a model this engine cannot load (missing, unfinished, corrupt, not the promoted one)."""


class LocalSource:
    kind = "local"

    def __init__(self, cfg: ServeConfig):
        self.cfg = cfg

    def wanted(self) -> Target:
        d = self.cfg.models                                   # re-reads production.json (MODEL_DIR overrides)
        return Target(key=str(d), run_id=None, model_dir=d)

    def materialize(self, t: Target) -> dict:
        return {"model_dir": t.model_dir}                     # Recommender works out run id + gold as before


class S3Source:
    kind = "s3"

    def __init__(self, cfg: ServeConfig, registry=None, cache_dir: Path | None = None):
        from pipeline.pointer import S3Pointer
        from pipeline.storage import Registry, S3Config
        if registry is None:
            s3 = S3Config.from_env()
            if s3 is None:
                raise RuntimeError("MODEL_REGISTRY=s3 needs S3_ENDPOINT_URL, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY")
            registry = Registry(s3, cfg.category, os.environ.get("MODEL_NAMESPACE") or None)
        self.reg, self.pointer = registry, S3Pointer(registry)
        self.cache = Path(cache_dir or os.environ.get("MODEL_CACHE_DIR", ".local/registry-cache")).resolve()

    def wanted(self) -> Target:
        p = self.pointer.read()                               # one small GET per poll
        if p is None:
            raise ModelUnavailable(f"no production pointer at s3://{self.reg.cfg.bucket}/{self.pointer.key}")
        return Target(key=f"{p['run_id']}@{p.get('manifest_sha256', '')}", run_id=p["run_id"],
                      version=p.get("version"), entry=p)

    def materialize(self, t: Target) -> dict:
        from pipeline.pointer import manifest_sha256
        from pipeline.storage import IntegrityError
        m = self.reg.run_manifest(t.run_id)
        if m is None:
            raise ModelUnavailable(f"run {t.run_id} has no MANIFEST.json in the registry (unfinished upload?)")
        want = t.entry.get("manifest_sha256")
        if want and manifest_sha256(m) != want:
            raise ModelUnavailable(f"run {t.run_id}: the registry's manifest is not the one that was promoted")
        try:
            model_dir, gold_dir = self.reg.fetch_run(t.run_id, self.cache)
        except IntegrityError as e:
            raise ModelUnavailable(str(e)) from None
        return {"model_dir": model_dir, "gold_dir": gold_dir, "run_id": t.run_id}


def pipeline_paths(cfg: ServeConfig):
    """pipeline Paths for the pointer THIS engine follows (admin rollback, production history): same category and,
    in S3 mode, the same MODEL_NAMESPACE, so a demo engine never rolls back the real pointer."""
    from pipeline.config import Paths, load_config
    sets = [f"category={cfg.category}"]
    ns = os.environ.get("MODEL_NAMESPACE")
    if ns and os.environ.get("MODEL_REGISTRY", "local").lower() == "s3":
        sets.append(f"namespace={ns}")
    return Paths(load_config(None, sets))


def model_source(cfg: ServeConfig):
    mode = os.environ.get("MODEL_REGISTRY", "local").lower()
    if mode == "local":
        return LocalSource(cfg)
    if mode == "s3":
        return S3Source(cfg)
    raise ValueError(f"MODEL_REGISTRY must be 'local' or 's3', got {mode!r}")
