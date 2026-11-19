# P2 roadmap: model registry on S3 (SeaweedFS)

Status: started 2026-11-19. **Phase A done**: the registry code, a local SeaweedFS, CI on a real S3 server.

Phase A (2026-11-19), verified on the laptop: the production run `20261111-171751` (10 files, 60.9 MB) and its
gold snapshot `35f9040f573514ae` (8 files, 54.2 MB) published to SeaweedFS and fetched back byte-identical
(`filecmp`, 18/18 files); a second fetch reused the verified cache, a second publish was a no-op. Single runs on
localhost: publish 4.2 s, fetch 1.5 s, cached fetch 0.5 s. A tampered object is refused and nothing is cached
(moto + real server); a run without MANIFEST (crashed upload) is never fetched. SeaweedFS binds 127.0.0.1 only and
answers 403 without credentials.
 Fixes gap **G1** (models live on one disk; a second engine host has none; promotion
means copying files by hand). Definition of done (ARCHITECTURE.md §7): *promote on one machine → two engine
replicas swap with 0 failed requests under load.*

## Decisions (approved 2026-11-19)
- **Server: SeaweedFS** (Apache-2.0, single `weed` binary, official Windows build), not MinIO: the MinIO
  community repo is archived (GitHub API `archived: true`) and ships no binaries; MinIO AIStor Free is
  proprietary, needs a license key and forbids redistribution. Pinned **4.52**, md5 checked, in the git-ignored
  `.local/seaweedfs/` (binary + data on E:). CI runs the same version as a container.
- **Client: boto3** with `endpoint_url`: the code speaks the S3 protocol only, so AWS S3 or MinIO AIStor later is
  a config change. Unit tests use **moto** (in-process S3); integration tests use the real server.
- Only production-history runs are published; `replay/` and `studies/` (research, ~1.7 GB) stay local.
- Local-disk mode stays (dev, tests, CI smoke): S3 mode is on only when `S3_ENDPOINT_URL` is set.

## Bucket layout (ADR-3, extended)
```
s3://recosphere/
├── models/<cat>/runs/<run_id>/            immutable; MANIFEST.json written LAST = the run is complete
│   ├── two_tower.pt  item_emb.npy  faiss.index  faiss_meta.json  ranker.txt  ranker_meta.json
│   ├── serving_policy.json  metrics.json  gate_report.json  config.yaml
│   └── MANIFEST.json                     {run_id, files: {name: {sha256, size}}, gold: <gold_hash>, created_at}
├── data/<cat>/gold/<gold_hash>/           immutable gold snapshot the engine reads with the model
│   ├── train/val/test.parquet  user_map.parquet  item_features.parquet  item_text_emb_<enc>.npy
│   ├── item_types.parquet (when present)  item_map.parquet
│   └── MANIFEST.json                     gold_hash = sha256 over (name, sha256) of those files
└── models/<cat>/production.json           pointer {run_id, version, manifest_sha256, gold, reason, promoted_at, metrics}
    models/<cat>/history/<version>.json    immutable copy of every pointer (audit + rollback target list)
```
Why the gold snapshot: the engine reads `data/gold/<cat>/` (history, user map, item features, text
embeddings) next to the model files, and a model is only correct with the data it was trained on. The run's
`data_hash` covers only its train/val/test split, so the catalog files get their own content hash.

## Phases (one commit each, CI green, ask before the next)
| | Builds | Done when |
|---|---|---|
| **A** | `pipeline/storage.py` (client from env, sha256 manifests, publish/fetch run + gold, verify), `scripts/s3_local.sh` (+ `make s3-up`), moto unit tests, real-server test, CI service | a real run round-trips through SeaweedFS byte-identical; a corrupted object is refused |
| **B** | pointer + history on S3: `promote`/`rollback` write `production.json` with a version counter; local mode unchanged | gate tests green in both modes; rollback picks the previous version |
| **C** | `serve/model_store.py`: poll pointer → download to cache → verify → hot swap (existing 7d logic); `ServeConfig` S3 mode | a corrupt or half-uploaded run is never served; the engine keeps the old model |
| **D** | `pipeline.run` publishes + promotes to S3; engine admin rollback on S3; migrate the 2 production-history runs | the real engine serves the real model from SeaweedFS |
| **E** | two engine replicas + locust load + promote | both swap, 0 failed requests (measured, numbers in REPORT) |

## Files
```
pipeline/storage.py            🆕 A  S3 client, manifests, publish/fetch run + gold, verify
pipeline/gate.py               ✏️ B  promote/rollback through a pointer store (local | s3)
pipeline/run.py                ✏️ D  publish after the gate
serve/model_store.py           🆕 C  pointer poll, download, cache, verify
serve/config.py, model_manager.py ✏️ C  S3 mode
scripts/s3_local.sh            🆕 A  start SeaweedFS (weed mini) on 127.0.0.1 with credentials from .env
Makefile                       ✏️ A  s3-up, s3-publish
requirements-serve.txt         ✏️ A  boto3; requirements-ci.txt: moto[s3]
.env.example                   ✏️ A  S3_ENDPOINT_URL, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY, S3_BUCKET
.github/workflows/ci.yml       ✏️ A  SeaweedFS container for the integration test
tests/test_storage.py          🆕 A  moto unit tests + real-server test (skipped without S3_ENDPOINT_URL)
docs/platform/P2_ROADMAP.md    🆕    this file
```
