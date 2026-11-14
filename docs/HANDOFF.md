# Handoff: state of RecoSphere (2026-11-14)

Context moved from a claude.ai session to Claude Code in VS Code. This file is the bridge.

## Done
| Area | State | Key numbers (real data unless marked) |
|---|---|---|
| Data, baselines, two-tower, FAISS, ranker, re-rank | ✅ notebooks 01–06 | ranker R@100 0.1348, NDCG@10 0.0184 (notebook); pipeline 3 seeds NDCG@10 0.0202 ± 0.0008 |
| Serving | ✅ FastAPI + Redis, parity 300/300, 7.7 ms/request | ~430 RPS per worker (GIL-bound → multi-worker) |
| Pipeline + MLflow + gate + promote/rollback | ✅ | gate CI half-width: NDCG@10 ±5.2%, R@100 ±1.7% |
| Hot swap | ✅ | synthetic: 0 errors across swaps under load |
| Ranker early stopping fix (3-fold mean curve) | ✅ default `ranker.cv_folds: 3` | single 0.0206 ± 0.0008 vs cv3 0.0210 ± 0.0008 (5 seeds); robustness fix, gain not established |
| Time-travel replay | ✅ | 90-day-old model −11.5% NDCG@10 vs fresh; never retrain −13.5% vs quarterly |
| A/B framework | ✅ | real data, 91k simulated users: no-slots vs slots clicks −0.8% [−6.9%, +5.4%], new-item share 12.4% → 2.6% → DON'T SHIP |
| Monitoring, Airflow DAG, CI | ✅ code; Docker stack + GitHub CI not yet verified on the laptop |
| RecoSphere Play (web/) | ✅ built + e2e tested on synthetic engine data; **not yet run on real data, not yet pushed** |
| Platform architecture | ✅ `docs/platform/ARCHITECTURE.md` (gap analysis, ADR-1..11, P0–P10) |

## Not yet in GitHub (as of commit 112762e)
- `web/` (whole app), `serve/product.py` and the engine changes for the app, `docs/platform/`,
  `CLAUDE.md`, this file.
- **`data/*.py`** (ingestion code) is missing on GitHub because `.gitignore` had `data/`. Fixed in the
  new `.gitignore` (`/data/*` + `!/data/*.py`). Commit your local `data/*.py` after applying it.

## Next: Phase P0 (foundation): checklist in docs/platform/REPO_STRUCTURE.md
1. Apply the new `.gitignore`, commit `data/*.py`, `web/`, the engine changes, docs. Push. CI must be green
   (jobs: test, web).
2. Run RecoSphere Play on the REAL engine data once (engine + `cd web; npm run dev`) and look at "Why this?"
   and "Because you played" with real titles.
3. Repo layout + `Makefile`/`justfile` + compose profiles (`core`, `stream`, `ml`, `obs`) so that one
   command starts the current system.
4. `docker compose up --build` once on the laptop (never verified there yet).

Then P1 (Postgres) → P2 (MinIO model registry) → P3 (Redpanda events + lake) → P4 (real time).

## Open decisions
- Cloud for P10: AWS (original plan) or GCP.
- Local GPU in Docker for daily training (WSL2 + NVIDIA Container Toolkit) or CPU containers.

## Files from the claude.ai session that are useful references
- `REPORT.md`: full technical report incl. the incident table
- `docs/AB_TESTING.md`: A/B concepts (Hinglish) and results
- Interview round 1 questions (16, L1–L5): kept outside the repo (PDF). Answers not yet submitted.
