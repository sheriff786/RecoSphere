# RecoSphere (RankForge)

A production-style multi-stage recommender, built end to end from raw data to monitored, A/B-testable
serving. It runs on Amazon Reviews 2023 (`Video_Games`: 814k interactions, 95k users, 25.6k games after
5-core).

```
raw .jsonl ─▶ PySpark bronze/silver/gold ─▶ two-tower (ID + MiniLM text) ─▶ FAISS top-200 ┐
                                             recent popularity top-100 ────────────────────┴▶ LightGBM LambdaRank (22 features)
                                                                                                ─▶ 2 new-item slots ─▶ top-10
offline:  pipeline.run (train → evaluate → paired-bootstrap gate → promote) · MLflow · time-travel replay · Airflow DAG
online:   FastAPI + Redis · hot model swap · A/B assignment + exposure/click logs · Prometheus/Grafana
```

## Results (test window Jul 2021 → Sep 2023, 21,653 users)

| Model | Recall@100 | NDCG@10 |
|---|---|---|
| EASE | 0.0444 | 0.0075 |
| Recent popularity | 0.0916 | 0.0102 |
| Two-tower (ID + text) | 0.0975 | 0.0146 |
| **Two-tower + popularity candidates → LightGBM ranker** | **0.1348** | **0.0184** |
| Same, pipeline, 3 seeds | 0.1347 ± 0.0027 | 0.0202 ± 0.0008 |

- 7.7 ms per request on a laptop CPU. Training/serving parity: 300/300 users.
- Paired bootstrap gate: on NDCG@10 the 95% CI is ±5.2%, measured.
- A model goes stale fast: after 90 days it is −11.5% NDCG@10 behind a fresh model, and never retraining
  loses 13.5% vs quarterly retraining (6-window time-travel replay).
- A/B framework with simulated users (91k), real API and logs: removing the new-item slots did not change
  clicks (−0.8%, CI [−6.9%, +5.4%]) but cut new-item exposure 12.4% → 2.6%, so the guardrails rejected it.
  A 5k-user subsample had shown a false "+35%, p = 0.02" (type-M error).

Full write-up with every number, method and incident: **[REPORT.md](REPORT.md)**.

## Repository

| Path | What |
|---|---|
| `notebooks/01–06` | research: ingestion, split, two-tower, ANN, ranker, beyond-accuracy |
| `baselines/`, `eval/metrics.py` | baselines; one metric implementation used everywhere |
| `retrieval/two_tower.py`, `ranking/pipeline.py` | model + feature code shared by training and serving (no skew) |
| `pipeline/` | `run.py` (steps), `gate.py` (bootstrap gate, promote, rollback), `tracking.py` (MLflow), `replay.py` (time travel) |
| `serve/` | FastAPI app, feature store (memory/Redis), hot swap, experiments, `/metrics`, parity check |
| `ab/` | traffic simulator, statistics (SRM, CUPED, power, peeking, interleaving), analysis report ([docs/AB_TESTING.md](docs/AB_TESTING.md)) |
| `monitoring/` | Prometheus config, alert rules, Grafana dashboard |
| `orchestration/` | Airflow DAG + Windows Task Scheduler script ([README](orchestration/README.md)) |
| `scripts/` | synthetic data, CI smoke test, studies, one-off repairs |
| `tests/`, `.github/workflows/ci.yml` | unit tests + end-to-end smoke test on synthetic data |

## Quick start

```bash
pip install -r requirements.txt
bash scripts/ci_smoke.sh             # whole system on generated synthetic data (~1 min), no downloads

# real data (needs Java for the Spark notebooks 01-02, a GPU helps for the two-tower)
python -m pipeline.run               # train → gate → promote (MLflow: mlflow ui --backend-store-uri sqlite:///mlflow.db)
uvicorn serve.app:app --port 8000    # GET /recommend/<user_id>, /health, /metrics
docker compose up --build            # API + Redis + Prometheus + Grafana (localhost:3000)
```

Docs: [REPORT.md](REPORT.md) · [pipeline](pipeline/README.md) · [serving](serve/README.md) ·
[A/B testing](docs/AB_TESTING.md) · [monitoring + CI](monitoring/README.md) · [orchestration](orchestration/README.md)
