# Handoff: state of RecoSphere (2026-11-14, after P0)

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
| Monitoring, Airflow DAG, CI | ✅ GitHub CI green (jobs: test, web, e2e); Prometheus/Grafana (`make up-all`) not yet run on the laptop |
| RecoSphere Play (web/) | ✅ pushed; runs on the real model in Docker; Playwright e2e in CI (synthetic) and `make e2e` (real) |
| Platform architecture | ✅ `docs/platform/ARCHITECTURE.md` (gap analysis, ADR-1..11, P0–P10) |
| **P0 foundation** | ✅ 2026-11-14 | `infra/` layout, `serve/routers/*`, `Makefile` + `.env.example` + compose profiles, Playwright e2e; `make up` on the laptop: real model, 94,762 users, 25,612 games |

## P0 findings to remember
- New-item slots are filled only from the ranker's top-100 candidates, so a user with no new game there gets
  0 new items (seen on a real app user: 2 of 3 requests had n_new=0). By design; revisit with P6 (new-game embedding).
- The web Docker image had never been built (missing `web/public`); fixed in P0.
- The app returned 500 on every page while the engine loaded its models; compose now waits for a healthy engine.
- Laptop (16 GB): do not train while the real `make up` stack runs (paging file runs out). See CLAUDE.md rules.

## Next: Phase P1 (Postgres)
Concept first, then code (ARCHITECTURE.md §7). Scope: `web/lib/db.ts` SQLite → Postgres, `web/db/migrations/`,
Postgres in the `core` compose profile, `tests/integration/` against the compose stack, e2e stays green.

Then P2 (MinIO model registry) → P3 (Redpanda events + lake) → P4 (real time).

## Open decisions
- Cloud for P10: AWS (original plan) or GCP.
- Local GPU in Docker for daily training (WSL2 + NVIDIA Container Toolkit) or CPU containers.

## Files from the claude.ai session that are useful references
- `REPORT.md`: full technical report incl. the incident table
- `docs/AB_TESTING.md`: A/B concepts (Hinglish) and results
- Interview round 1 questions (16, L1–L5): kept outside the repo (PDF). Answers not yet submitted.
