"""
Model registry on S3 (P2, docs/platform/P2_ROADMAP.md). Any S3 server works: SeaweedFS locally and in CI,
AWS S3 or MinIO later; only the endpoint and keys change (`S3_*` env vars, see .env.example).

A run is an immutable prefix models/<cat>/runs/<run_id>/ holding the files the engine serves plus
MANIFEST.json = {run_id, files: {name: {sha256, size}}, gold, created_at}. MANIFEST is written LAST, so a run
without one is an unfinished upload and is never fetched. The engine also reads the gold catalog files
(history, user map, item features, text embeddings) and a model is only right with the data it was trained on,
so the gold snapshot is published too, under its own content hash: data/<cat>/gold/<gold_hash>/.

fetch_* download into a temporary folder, check every file's size + sha256 against the manifest, and only then
rename the folder into the cache. A corrupt or partial download is never left where a loader could find it.

    python -m pipeline.storage publish-run <run_id>        # run + its gold snapshot -> S3
    python -m pipeline.storage fetch-run <run_id> <dir>    # -> <dir>/runs/<run_id>, verified
    python -m pipeline.storage ls
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

RUN_FILES = ["two_tower.pt", "item_emb.npy", "faiss.index", "faiss_meta.json", "ranker.txt", "ranker_meta.json",
             "serving_policy.json"]                      # what the engine loads (serve/recommender.py)
RUN_OPTIONAL = ["metrics.json", "gate_report.json", "config.yaml",   # provenance, small
                "data/manifest.json"]                    # the run's time windows: the gate's leakage guard reads it
GOLD_FILES = ["train.parquet", "val.parquet", "test.parquet", "user_map.parquet", "item_map.parquet",
              "item_features.parquet"]
GOLD_OPTIONAL = ["item_types.parquet"]                   # + every item_text_emb_<encoder>.npy
MANIFEST = "MANIFEST.json"


class IntegrityError(RuntimeError):
    """A downloaded file does not match its manifest (size or sha256), or a manifest is missing."""


@dataclass(frozen=True)
class S3Config:
    endpoint_url: str | None
    access_key: str
    secret_key: str
    bucket: str
    region: str = "us-east-1"

    @classmethod
    def from_env(cls) -> "S3Config | None":
        """None when S3_ENDPOINT_URL is unset: the pipeline and the engine stay in local-disk mode."""
        url = os.environ.get("S3_ENDPOINT_URL")
        if not url:
            return None
        return cls(endpoint_url=url, access_key=os.environ["S3_ACCESS_KEY_ID"],
                   secret_key=os.environ["S3_SECRET_ACCESS_KEY"], bucket=os.environ.get("S3_BUCKET", "recosphere"),
                   region=os.environ.get("S3_REGION", "us-east-1"))

    def client(self):
        import boto3
        from botocore.config import Config
        return boto3.client("s3", endpoint_url=self.endpoint_url, aws_access_key_id=self.access_key,
                            aws_secret_access_key=self.secret_key, region_name=self.region,
                            config=Config(signature_version="s3v4", s3={"addressing_style": "path"},
                                          retries={"max_attempts": 5, "mode": "standard"}))


def load_s3_env(path: str | Path = ".env") -> None:
    """For the CLIs: take S3_* from the repo's .env when the environment does not set them. Only S3_* lines are
    read (the file holds other secrets), and a value already in the environment always wins."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        key, sep, val = line.strip().partition("=")
        if sep and key.startswith("S3_") and key not in os.environ:
            os.environ[key] = val.strip().strip('"').strip("'")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _describe(folder: Path, names: list[str]) -> dict:
    return {n: {"sha256": sha256_file(folder / n), "size": (folder / n).stat().st_size} for n in names}


def gold_files(gold_dir: Path) -> list[str]:
    missing = [n for n in GOLD_FILES if not (gold_dir / n).exists()]
    if missing:
        raise FileNotFoundError(f"missing in {gold_dir}: {missing}")
    extra = [n for n in GOLD_OPTIONAL if (gold_dir / n).exists()]
    emb = sorted(p.name for p in gold_dir.glob("item_text_emb_*.npy"))
    return GOLD_FILES + extra + emb


def gold_hash(files: dict) -> str:
    """Content hash of a gold snapshot: the same files with the same bytes -> the same hash, on any machine."""
    h = hashlib.sha256()
    for name in sorted(files):
        h.update(f"{name}:{files[name]['sha256']}\n".encode())
    return h.hexdigest()[:16]


class Registry:
    def __init__(self, cfg: S3Config, category: str, namespace: str | None = None, s3=None):
        self.cfg, self.s3 = cfg, s3 or cfg.client()
        ns = f"/{namespace.strip('/')}" if namespace else ""      # replays / tests never touch real runs or gold
        self.models = f"models/{category}{ns}"
        self.gold = f"data/{category}{ns}/gold"

    # ------------------------------------------------------------------ keys
    def run_prefix(self, run_id: str) -> str:
        return f"{self.models}/runs/{run_id}"

    def gold_prefix(self, h: str) -> str:
        return f"{self.gold}/{h}"

    # ------------------------------------------------------------------ low level
    def ensure_bucket(self):
        from botocore.exceptions import ClientError
        try:
            self.s3.head_bucket(Bucket=self.cfg.bucket)
        except ClientError:
            self.s3.create_bucket(Bucket=self.cfg.bucket)

    def get_json(self, key: str) -> dict | None:
        from botocore.exceptions import ClientError
        try:
            body = self.s3.get_object(Bucket=self.cfg.bucket, Key=key)["Body"].read()
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404", "NotFound"):
                return None
            raise
        return json.loads(body)

    def put_json(self, key: str, obj: dict):
        self.s3.put_object(Bucket=self.cfg.bucket, Key=key, Body=json.dumps(obj, indent=2).encode(),
                           ContentType="application/json")

    def _upload(self, folder: Path, prefix: str, files: dict):
        for name in files:
            self.s3.upload_file(str(folder / name), self.cfg.bucket, f"{prefix}/{name}")

    def _download(self, prefix: str, manifest: dict, dest: Path) -> Path:
        """Download every manifest file into a temp folder next to `dest`, verify, then rename into place."""
        if (dest / MANIFEST).exists() and json.loads((dest / MANIFEST).read_text()) == manifest:
            return dest                                            # cached and verified earlier
        tmp = dest.parent / f".{dest.name}.tmp-{uuid.uuid4().hex[:8]}"
        tmp.mkdir(parents=True)
        try:
            for name, want in manifest["files"].items():
                (tmp / name).parent.mkdir(parents=True, exist_ok=True)      # data/manifest.json
                self.s3.download_file(self.cfg.bucket, f"{prefix}/{name}", str(tmp / name))
                size, digest = (tmp / name).stat().st_size, sha256_file(tmp / name)
                if size != want["size"] or digest != want["sha256"]:
                    raise IntegrityError(f"{prefix}/{name}: got {size} bytes sha256 {digest[:12]}, manifest says "
                                         f"{want['size']} bytes sha256 {want['sha256'][:12]}")
            (tmp / MANIFEST).write_text(json.dumps(manifest, indent=2))   # written last: marks a verified copy
            if dest.exists():
                shutil.rmtree(dest)
            tmp.rename(dest)
            return dest
        finally:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)

    # ------------------------------------------------------------------ gold
    def publish_gold(self, gold_dir: Path, log=print) -> str:
        files = _describe(gold_dir, gold_files(gold_dir))
        h = gold_hash(files)
        if self.get_json(f"{self.gold_prefix(h)}/{MANIFEST}") is not None:
            log(f"gold {h}: already published")
            return h
        self._upload(gold_dir, self.gold_prefix(h), files)
        self.put_json(f"{self.gold_prefix(h)}/{MANIFEST}",
                      {"gold": h, "files": files, "created_at": datetime.now(timezone.utc).isoformat()})
        log(f"gold {h}: published {len(files)} files, {sum(f['size'] for f in files.values()) / 1e6:.1f} MB")
        return h

    def fetch_gold(self, h: str, cache_root: Path) -> Path:
        m = self.get_json(f"{self.gold_prefix(h)}/{MANIFEST}")
        if m is None:
            raise IntegrityError(f"gold snapshot {h} has no manifest (missing or unfinished upload)")
        return self._download(self.gold_prefix(h), m, Path(cache_root) / "gold" / h)

    # ------------------------------------------------------------------ runs
    def run_manifest(self, run_id: str) -> dict | None:
        return self.get_json(f"{self.run_prefix(run_id)}/{MANIFEST}")

    def publish_run(self, run_dir: Path, gold_dir: Path, log=print) -> dict:
        """Upload a run's serving files + its gold snapshot. Runs are immutable: publishing the same run again is a
        no-op when the bytes match and an error when they differ."""
        run_dir = Path(run_dir)
        missing = [n for n in RUN_FILES if not (run_dir / n).exists()]
        if missing:
            raise FileNotFoundError(f"{run_dir}: missing {missing}")
        files = _describe(run_dir, RUN_FILES + [n for n in RUN_OPTIONAL if (run_dir / n).exists()])
        g = self.publish_gold(gold_dir, log)
        old = self.run_manifest(run_dir.name)
        if old is not None:
            if old["files"] == files and old["gold"] == g:
                log(f"run {run_dir.name}: already published")
                return old
            raise IntegrityError(f"run {run_dir.name} is already published with different files: runs are immutable")
        self._upload(run_dir, self.run_prefix(run_dir.name), files)
        manifest = {"run_id": run_dir.name, "files": files, "gold": g,
                    "created_at": datetime.now(timezone.utc).isoformat()}
        self.put_json(f"{self.run_prefix(run_dir.name)}/{MANIFEST}", manifest)
        log(f"run {run_dir.name}: published {len(files)} files, {sum(f['size'] for f in files.values()) / 1e6:.1f} MB"
            f" (gold {g})")
        return manifest

    def fetch_run(self, run_id: str, cache_root: Path) -> tuple[Path, Path]:
        """-> (model dir, gold dir), both verified against their manifests."""
        m = self.run_manifest(run_id)
        if m is None:
            raise IntegrityError(f"run {run_id} has no manifest (missing or unfinished upload)")
        gold = self.fetch_gold(m["gold"], cache_root)
        return self._download(self.run_prefix(run_id), m, Path(cache_root) / "runs" / run_id), gold

    def list_runs(self) -> list[str]:
        out, token = [], None
        while True:
            kw = {"Bucket": self.cfg.bucket, "Prefix": f"{self.models}/runs/", "Delimiter": "/"}
            if token:
                kw["ContinuationToken"] = token
            r = self.s3.list_objects_v2(**kw)
            out += [p["Prefix"].rstrip("/").rsplit("/", 1)[-1] for p in r.get("CommonPrefixes", [])]
            if not r.get("IsTruncated"):
                return out
            token = r["NextContinuationToken"]


def main(argv=None):
    import argparse
    from pipeline.config import Paths, load_config
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["publish-run", "fetch-run", "ls"])
    ap.add_argument("run_id", nargs="?")
    ap.add_argument("dest", nargs="?")
    ap.add_argument("--set", action="append", default=[], help="config override, e.g. category=Video_Games")
    a = ap.parse_args(argv)
    load_s3_env()
    cfg = S3Config.from_env()
    if cfg is None:
        raise SystemExit("S3_ENDPOINT_URL is not set (see .env.example)")
    paths = Paths(load_config(None, a.set))
    reg = Registry(cfg, paths.category)
    if a.cmd == "ls":
        print("\n".join(reg.list_runs()) or "(no runs)")
    elif a.cmd == "publish-run":
        reg.ensure_bucket()
        reg.publish_run(paths.runs / a.run_id, paths.gold)
    else:
        model, gold = reg.fetch_run(a.run_id, Path(a.dest))
        print(f"model {model}\ngold  {gold}")


if __name__ == "__main__":
    main()
