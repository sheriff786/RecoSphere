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

Real results and every incident: `REPORT.md`. A/B method: `docs/AB_TESTING.md`.

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
| `web/` | Next.js 15 app: auth (SQLite now), onboarding, rows, "Why this?" (SHAP), stream player, admin |
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
cd web; npm run dev                                   # app on :3000 (needs .env.local from .env.example)
bash scripts/ci_smoke.sh                              # whole system on synthetic data (WSL / Git Bash)
make up / up-all / down / config / help               # Docker stack (Git Bash); config: .env from .env.example
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
- **Web:** the browser never calls the engine directly; logout navigates client-side (a server redirect
  sent users to `localhost` and lost the cookie).
- **.gitignore:** the root has `lib/` (Python template), so `!web/lib/` must stay. `data/` FILES are
  ignored, `data/*.py` are not.
- Kill Next with `next-server`, not `next start` (the process name differs), or old code keeps serving.

## Next
See `docs/HANDOFF.md` → Phase **P0 (foundation)**. Target layout + P0 checklist: `docs/platform/REPO_STRUCTURE.md`.
Build cycle per phase: concept (Hinglish) → code → unit tests → integration test → automated demo → commit + CI green.
