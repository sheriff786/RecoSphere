# Target repo structure and build workflow

Agreed on 2026-11-14 (see ARCHITECTURE.md §7 for the phases). Legend: ✅ exists · ✏️ changes · 🆕 new (phase).

**Decision: existing Python packages stay at the root.** `retrieval/`, `ranking/`, `eval/`, `pipeline/` and
`serve/` are imported by both training and serving; that shared import is what keeps training/serving
parity (300/300). Moving them would break imports, CI and notebooks for no functional gain. New services
get new folders; everything that deploys goes under `infra/`.

```
RecoSphere/
├── CLAUDE.md  README.md  REPORT.md                 ✅
├── Makefile                                        🆕 P0  make up / up-all / test / e2e / down
├── .env.example                                    🆕 P0  every config value of every service
├── docs/
│   ├── platform/ARCHITECTURE.md, REPO_STRUCTURE.md ✅
│   ├── adr/                                        🆕 P0  ADR-1..11 as separate files
│   └── HANDOFF.md, AB_TESTING.md                   ✅
├── notebooks/  data/*.py                           ✅  (data FILES git-ignored, data/*.py committed)
├── eval/  retrieval/  ranking/  baselines/         ✅  shared ML core
├── pipeline/                                       ✅
│   ├── storage.py                                  🆕 P2  publish run to S3/MinIO, versioned pointer + checksum
│   └── lake.py                                     🆕 P5  training data from the event lake
├── serve/                                          ✅  ENGINE (FastAPI)
│   ├── app.py                                      ✏️ P0  thin: create app, wire routers, lifespan
│   ├── routers/                                    🆕 P0  recommend.py, catalog.py, events.py, admin.py
│   ├── model_store.py                              🆕 P2  pull from S3 → cache → verify checksum → hot swap
│   ├── events.py                                   🆕 P3  Kafka producer + retry buffer (ADR-5)
│   ├── clock.py                                    🆕 P4  real time instead of the frozen t_ref
│   └── item_state.py                               🆕 P4  popularity windows from Redis
├── contracts/events/                               🆕 P3  exposure.v1.json, feedback.v1.json, play.v1.json, catalog.v1.json
├── stream/                                         🆕 P3  STREAM-WORKER service
│   ├── main.py                                     consumer loop, at-least-once, dedup on event_id
│   ├── lake_sink.py                                🆕 P3  events → date-partitioned Parquet on MinIO
│   ├── popularity.py                               🆕 P4  30/90/365-day + trend windows → Redis
│   ├── catalog_embed.py                            🆕 P6  new game → MiniLM → item tower → index delta
│   └── Dockerfile
├── ab/                                             ✅  ✏️ P3 analyze.py reads the lake
├── web/                                            ✅  RecoSphere Play (Next.js 15)
│   ├── lib/db.ts                                   ✏️ P1  SQLite → Postgres
│   ├── db/migrations/                              🆕 P1  001_users.sql, 002_plays.sql ...
│   └── e2e/                                        🆕 P0  Playwright: signup → onboarding → play → feed changes → admin
├── infra/                                          🆕 P0
│   ├── compose/docker-compose.yml                  ✏️ P0  moved from root; profiles core | stream | ml | obs
│   ├── compose/{postgres,minio,redpanda,traefik}/  🆕 P1/P2/P3/P7  init scripts, config
│   ├── monitoring/                                 ✏️ P0  moved from monitoring/ (+Tempo/Loki P8)
│   ├── airflow/                                    ✏️ P0  moved from orchestration/
│   ├── helm/                                       🆕 P9
│   └── terraform/                                  🆕 P10
├── tests/  (+ tests/integration/ 🆕 P1, runs against the compose stack)
├── scripts/                                        ✅
└── .github/workflows/ci.yml ✏️  images.yml 🆕 P0
```

## How every phase is built
1. Concept first (Hinglish, a diagram) → 2. code → 3. unit tests → 4. integration test on the compose
stack → 5. demo = the phase's definition of done (automated) → 6. commit, CI green → next phase.
"Works on my machine" is not done.

## P0 checklist
- [x] `infra/`: move `docker-compose.yml`, `monitoring/`, `orchestration/` (update CI + docs paths)
- [x] split `serve/app.py` into `serve/routers/*` (no behaviour change; all tests + smoke test pass)
- [ ] `Makefile`, `.env.example`, compose profiles (`make up` = core: web, engine, redis)
- [ ] `web/e2e/` Playwright test + `make e2e`; run it in CI against the synthetic engine
- [ ] Done = `make up` → app + engine run; `make e2e` passes; GitHub CI green (jobs: test, web, e2e)

Local prerequisites: Docker Desktop (WSL2), Node 22, Python venv `.reco`, `make` (Git Bash/WSL or
`choco install make`). From P5: NVIDIA Container Toolkit in WSL2 for GPU training (or CPU).
