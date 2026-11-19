# CLAUDE.md: RecoSphere (RankForge)

Project memory for Claude Code. Read this first, then `docs/HANDOFF.md` (where we are, what is next) and
`docs/platform/ARCHITECTURE.md` (the plan every new phase follows).

## Working with Sheriff (the owner)
- Explain in **Hinglish**, concepts first, then code. Visuals, analogies and dry-run traces help.
- Be honest and adversarial. Call out flaws, including your own mistakes. No generic praise.
- **Never invent numbers.** Every metric must come from code run on the real data (or be labelled
  *synthetic*). Single-seed results are labelled; use mean ± std when seeds exist.
- He builds and self-hosts. Prefer open-source, self-hostable components over managed platforms.
- Repo docs (README, REPORT, docs/) are in **English** (recruiters read them); chat is Hinglish.
- Before claiming something works, run it. After a fix, re-run the test that found the bug.

## What this is
A production-style multi-stage recommender on Amazon Reviews 2023 `Video_Games` (814,586 interactions,
94,762 users, 25,612 games after 5-core), plus a product (RecoSphere Play, `web/`) on top of it.

```
data (PySpark notebooks 01-02) -> two-tower (ID + MiniLM text) -> FAISS -> + recent-pop candidates
-> LightGBM LambdaRank (22 features, 3-fold-CV tree count) -> 2 new-item slots -> top-10
offline: pipeline.run (train -> evaluate -> paired-bootstrap gate -> promote), MLflow, replay, Airflow DAG
online:  FastAPI engine (Redis, hot swap, A/B, /metrics, product API) <- Next.js app (web/)
```

Real results and every incident: `REPORT.md`. A/B method: `docs/AB_TESTING.md`. Every endpoint: `docs/API.md`.

## Layout
| Path | What |
|---|---|
| `notebooks/` | research phases 01-06 (Spark ingestion → beyond-accuracy) |
| `data/*.py` | ingestion/split code (the data FILES under `data/` are git-ignored, the `.py` files are not) |
| `eval/metrics.py` | the ONE metric implementation (recall/ndcg, warm/cold, per-user) |
| `retrieval/two_tower.py`, `ranking/pipeline.py` | model + feature code shared by training and serving (parity) |
| `pipeline/` | `run.py` steps, `gate.py` (bootstrap gate, promote, rollback), `replay.py`, `config.yaml` |
| `serve/` | engine: `app.py` routes, `recommender.py`, `feature_store.py`, `experiments.py`, `metrics.py`, `product.py` (app API), `parity_check.py` |
| `ab/` | simulator, stats (SRM, CUPED, power, peeking, interleaving), analyzer |
| `app_api/` | user data on Postgres (P1): `db/` (engine, pool), `models/` (SQLAlchemy), `schemas/`, `services/`, `routers/`, `migrations/` (Alembic) |
| `web/` | Next.js 15 app: auth + library + plays via app_api (`lib/api.ts`), onboarding, rows, "Why this?" (SHAP), stream player, admin |
| `infra/compose/`, `infra/monitoring/`, `infra/airflow/` | docker-compose, Prometheus/Grafana/alerts, Airflow DAG (everything that deploys) |
| `.github/workflows/ci.yml`, `scripts/` | CI, smoke test |
| `docs/platform/` | productionization architecture + phased plan (P0–P10) |

## Commands (Windows PowerShell, venv `.reco`)
```powershell
.reco\Scripts\Activate.ps1
python -m pytest -q                                   # unit tests
python -m pipeline.run                                # train -> gate -> promote (GPU auto-detected)
python -m pipeline.run --rollback
uvicorn serve.app:app --host 127.0.0.1 --port 8000    # engine; use 127.0.0.1, not localhost (IPv6 delay on Windows)
python -m serve.parity_check --n 300
python -m data.item_types --write                     # game/accessory/other per item -> gold/item_types.parquet (re-run after new data)
cd web; npm run dev                                   # app on :3000 (needs .env.local from .env.example)
bash scripts/ci_smoke.sh                              # whole system on synthetic data (WSL / Git Bash)
make up / up-all / down / config / help               # Docker stack (Git Bash); config: .env from .env.example
make app-api / db-upgrade / db-current / db-drift   # app_api on :8100; Alembic on the app DB; URLs in app_api/.env
make s3-up          # model registry: SeaweedFS 4.52 S3 on 127.0.0.1:8333 (binary + data in git-ignored .local/; S3_* in .env)
python -m pipeline.storage publish-run <run_id> | fetch-run <run_id> <dir> | ls   # P2 registry CLI
make load-catalog   # gold catalog -> Neon `games` (idempotent; run after new data, after `python -m data.item_types --write`)
ADMIN_PASSWORD=... python -m app_api.cli create-admin --email <email>   # first admin (or promote an account)
make e2e            # Playwright vs the running stack (no rollback)
E2E_PG_URL=<server url> make e2e-synthetic   # throwaway stack + throwaway DB (dropped), = CI job e2e
```

## Rules learned the hard way (do not regress)
- **Leakage:** popularity/recency features are computed as of the cutoff; 2023 metadata fields that change
  over time are excluded; the gate blocks a candidate whose test window predates production's.
- **Training/serving parity:** never re-implement a feature in serve/; use `ranking/pipeline.py`. Ties are
  broken by item id (`tie_safe_top`).
- **Replay/studies run in a namespace** (`--set namespace=replay/<name>`). `replay.py` refuses otherwise:
  a replay once overwrote the real `production.json`.
- **Gate:** paired bootstrap; block if CI_high < −1% or point drop > 5%. Gate policy comes from the
  CURRENT config even when continuing an old run.
- **Hot swap:** load in the background, smoke test, swap one reference; the feature store is not reloaded.
- **A/B:** assignment = salted sha256 by user; never decide when MDE > 10%; guardrails include coverage
  and new-item share; keep-alive HTTP in simulators (Windows port exhaustion).
- **Item types are a serving filter, not a feature:** the model still ranks accessories (it learned Amazon
  purchases); the app asks for `item_type=game` everywhere (`web/lib/engine.ts` defaults). Never add item_type
  to the ranker without retraining + the gate. Rules + audit: `data/item_types.py`.
- **App DB (DB_SCHEMA.md, P1_ROADMAP.md):** Neon is the main DB. Since P1 phase A the SQLAlchemy models in
  `app_api/models` are the schema: change a model -> `alembic revision --autogenerate` -> review -> `make db-upgrade`.
  Never edit tables by hand; `web/db/migrate.mjs` is frozen (Neon is stamped `0001_baseline`). Alembic uses the
  DIRECT host, the app the `-pooler` host; URLs only in `app_api/.env` / `web/.env.local`; CI uses Postgres 16.
  Pool: no pre-ping (it costs a round trip per request), `pool_recycle=240`; `prepare_threshold=None` for PgBouncer.
  `get_autocommit_db` only for reads and SINGLE-statement writes (pinned by a statement-count test); anything with
  two writes runs in `services/tx.transaction`. Services raise `services/errors.py`, never HTTP errors.
  Every app_api route except /health needs X-Service-Token (SERVICE_TOKEN in app_api/.env; fails closed when unset).
- **API doc:** any change that adds, changes or removes an engine route or a web `/api/*` route updates
  `docs/API.md` (table row + section + change log) in the same commit. `tests/test_api_doc.py` enforces it.
- **Web:** the browser never calls the engine or app_api directly; the cookie (JWT) carries only `sid`, liveness is
  app_api `/auth/me` (logout revokes). e2e and tests never write to the real Neon database (throwaway DBs / rollback);
  a real-stack run that creates users must delete them afterwards (`email like 'e2e-%@example.test'`).
- **Web (old rule):** the browser never calls the engine directly; logout navigates client-side (a server redirect
  sent users to `localhost` and lost the cookie).
- **.gitignore:** the root has `lib/` (Python template), so `!web/lib/` must stay. Same for `models/` (trained artifacts): `!app_api/models/` (SQLAlchemy code) must stay; check `git status` lists new source folders before committing. `data/` FILES are
  ignored, `data/*.py` are not.
- **E2E:** run the app bound to `0.0.0.0` (as in Docker). Bound to `127.0.0.1`, Next rewrites it to `localhost` in
  redirects (another origin). Rollback in e2e only with `E2E_ALLOW_ROLLBACK=1` on a throwaway stack.
- **Laptop (16 GB):** do not train (smoke/e2e-synthetic) while the real `make up` stack runs: paging file runs out
  (torch segfault / CUDA OOM). `make down` first. Rebuild one service with `up --no-deps <svc>` or compose
  recreates the engine too (1-2 min model load).
- **Model registry (P2):** S3 protocol only (boto3 + `endpoint_url`); SeaweedFS locally (MinIO community is archived).
  Runs and gold snapshots are immutable and content-checked (MANIFEST.json written last; fetch verifies sha256 before
  the folder appears). Tests use a namespace, never the real `models/<cat>/runs/` prefix. The pointer store is
  `MODEL_REGISTRY=local|s3` (pipeline/pointer.py); on S3 promotion is compare-and-swap (If-Match), never a blind write.
  The engine side is serve/model_store.py: demos use `MODEL_NAMESPACE` + their own `MODEL_CACHE_DIR`, deleted after.
- Kill Next with `next-server`, not `next start` (the process name differs), or old code keeps serving.
- **Engine history is a cache of Postgres `library`** (P1 phase F): the web replays it (`web/lib/history.ts`,
  engine `POST /users/{id}/history`). In a server component, Next memoizes identical GET fetches per render:
  re-fetching after a write returns the OLD answer; redirect (new request) instead.

## Next
P0 (foundation) is done. See `docs/HANDOFF.md` → Phase **P1 (Postgres)**. Target layout: `docs/platform/REPO_STRUCTURE.md`.
Build cycle per phase: concept (Hinglish) → code → unit tests → integration test → automated demo → commit + CI green.
