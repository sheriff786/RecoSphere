"""
Where "which model is production" lives (P2 phase B). Two stores, one contract:

  LocalPointer  models/<cat>/production.json + production_history.jsonl (Phase 7 behaviour, one machine)
  S3Pointer     s3://<bucket>/models/<cat>/production.json + history/<version>.json (any number of machines)

Every entry: {run_id, path, reason, promoted_at, metrics, version, ...}; `version` goes up by one per promotion.

S3 has no transactions, so the pointer object itself is the lock (compare-and-swap):
  1. read production.json and its ETag
  2. write the new pointer with If-Match: <ETag> (If-None-Match: * when there is none yet)
     -> if anyone promoted in between, S3 answers 412 and this promotion fails (ConcurrentPromotion)
  3. write history/<version>.json with If-None-Match: * (immutable)
A crash between 2 and 3 loses no promotion: history() always includes the current pointer.
Conditional writes are checked on SeaweedFS 4.52 and moto (both answer 412), and AWS S3 supports them.

Which store: MODEL_REGISTRY=s3 (needs S3_* env, see .env.example) or local (default). Opt-in on purpose: a
leftover S3_ENDPOINT_URL must not silently move the pipeline's production pointer somewhere else.
"""
from __future__ import annotations

import hashlib
import json
import os

from pipeline.storage import Registry, S3Config


class ConcurrentPromotion(RuntimeError):
    """Someone else moved the production pointer between our read and our write."""


def _status(e) -> int:
    return int(e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))


class LocalPointer:
    kind = "local"

    def __init__(self, paths):
        self.ptr, self.hist = paths.production_pointer, paths.production_history

    def read(self) -> dict | None:
        return json.loads(self.ptr.read_text()) if self.ptr.exists() else None

    def history(self) -> list[dict]:
        if not self.hist.exists():
            return []
        return [json.loads(x) for x in self.hist.read_text().splitlines() if x.strip()]

    def write(self, entry: dict) -> dict:
        cur = self.read()
        entry = {**entry, "version": int((cur or {}).get("version", len(self.history()))) + 1}
        self.ptr.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.ptr.with_suffix(".tmp")
        tmp.write_text(json.dumps(entry, indent=2))
        tmp.replace(self.ptr)                                   # atomic swap: a reader never sees half a file
        with self.hist.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        return entry


class S3Pointer:
    kind = "s3"

    def __init__(self, registry: Registry):
        self.reg = registry
        self.key = f"{registry.models}/production.json"
        self.hist_prefix = f"{registry.models}/history/"

    def _read_with_etag(self) -> tuple[dict | None, str | None]:
        from botocore.exceptions import ClientError
        try:
            r = self.reg.s3.get_object(Bucket=self.reg.cfg.bucket, Key=self.key)
        except ClientError as e:
            if _status(e) == 404 or e.response["Error"]["Code"] in ("NoSuchKey", "NotFound"):
                return None, None
            raise
        return json.loads(r["Body"].read()), r["ETag"]

    def read(self) -> dict | None:
        return self._read_with_etag()[0]

    def history(self) -> list[dict]:
        s3, b = self.reg.s3, self.reg.cfg.bucket
        keys, token = [], None
        while True:
            kw = {"Bucket": b, "Prefix": self.hist_prefix} | ({"ContinuationToken": token} if token else {})
            r = s3.list_objects_v2(**kw)
            keys += [o["Key"] for o in r.get("Contents", [])]
            if not r.get("IsTruncated"):
                break
            token = r["NextContinuationToken"]
        out = [json.loads(s3.get_object(Bucket=b, Key=k)["Body"].read()) for k in sorted(keys)]
        cur = self.read()
        if cur is not None and cur.get("version") not in {h.get("version") for h in out}:
            out.append(cur)                                      # crashed after the pointer, before history
        return sorted(out, key=lambda h: h.get("version", 0))

    def write(self, entry: dict) -> dict:
        from botocore.exceptions import ClientError
        cur, etag = self._read_with_etag()
        entry = {**entry, "version": int((cur or {}).get("version", 0)) + 1}
        body = json.dumps(entry, indent=2).encode()
        cond = {"IfMatch": etag} if etag else {"IfNoneMatch": "*"}
        try:
            self.reg.s3.put_object(Bucket=self.reg.cfg.bucket, Key=self.key, Body=body,
                                   ContentType="application/json", **cond)
        except ClientError as e:
            if _status(e) in (409, 412) or e.response["Error"]["Code"] in ("PreconditionFailed", "ConditionalRequestConflict"):
                raise ConcurrentPromotion(f"production pointer changed while promoting {entry['run_id']}; "
                                          "read it again and retry") from None
            raise
        try:
            self.reg.s3.put_object(Bucket=self.reg.cfg.bucket, Key=f"{self.hist_prefix}{entry['version']:08d}.json",
                                   Body=json.dumps(entry).encode(), ContentType="application/json", IfNoneMatch="*")
        except ClientError as e:
            if _status(e) != 412:                                 # 412 = already there (a retry): fine
                raise
        return entry


def manifest_sha256(manifest: dict) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def registry_for(paths) -> Registry:
    cfg = S3Config.from_env()
    if cfg is None:
        raise RuntimeError("MODEL_REGISTRY=s3 needs S3_ENDPOINT_URL, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY "
                           "(see .env.example)")
    return Registry(cfg, paths.category, getattr(paths, "namespace", None))


def pointer_store(paths):
    mode = os.environ.get("MODEL_REGISTRY", "local").lower()
    if mode == "local":
        return LocalPointer(paths)
    if mode == "s3":
        return S3Pointer(registry_for(paths))
    raise ValueError(f"MODEL_REGISTRY must be 'local' or 's3', got {mode!r}")
