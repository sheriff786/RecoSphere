# RecoSphere (RankForge): Technical Report

A production-style, multi-stage recommender built end to end on Amazon Reviews 2023 (`Video_Games`):
data lake → candidate generation → ranking → re-ranking → online serving → retraining pipeline with an
evaluation gate → zero-downtime model swaps → monitoring → A/B testing framework.

**Rule for this report:** every number is produced by code in this repo on the real dataset, unless it is
explicitly labelled *synthetic*. Single-seed results are marked as such; where seeds were run, mean ± std
is given.

---

## 1. Problem and data

Recommend 10 games to a user, given their purchase/review history, at <50 ms p95.

| | |
|---|---|
| Source | Amazon Reviews 2023 (McAuley Lab), category `Video_Games`, reviews + item metadata |
| Signal | implicit feedback: a review = the user acquired the game (ratings are not used as labels) |
| Raw → clean | 4,624,615 raw rows → 0 corrupt → 53,648 exact duplicates removed → 4,570,967 interactions |
| After dedup + iterative 5-core | **814,586 interactions, 94,762 users, 25,612 games** (matches the official 5-core benchmark exactly: 0 pairs differ) |
| Split | global temporal 80/10/10: train → 9 Dec 2019, val → 9 Jul 2021, test → Sep 2023 |

**The number that drove the design:** 46.2% of test interactions are on games that never appear in the
user-visible history (train+val). An ID-only model can never recommend them. → text-aware item tower,
popularity-recency candidates, new-item slots in re-ranking, frequent retraining.

**Leakage guards:** 2023 metadata snapshot fields that change over time (sales rank, rating counts, price)
are excluded; popularity features are computed only from interactions before each cutoff; training
histories are asserted to precede their targets.

## 2. Offline results (test window, 21,653 users with history, one shared metric implementation)

| Stage | Model | Recall@100 | NDCG@10 | Warm R@100 | Cold R@100 |
|---|---|---|---|---|---|
| Baseline | Popularity (all time) | 0.0239 | 0.0040 | 0.0379 | 0 |
| Baseline | EASE (λ=500) | 0.0444 | 0.0075 | 0.0702 | 0 |
| Baseline | **Recent popularity** | 0.0916 | 0.0102 | 0.1441 | 0 |
| Retrieval | Two-tower, ID + MiniLM text | 0.0975 | 0.0146 | 0.1327 | 0.0378 |
| Blend | RRF (untrained) | 0.1063 | 0.0168 | 0.1556 | 0.0213 |
| **Ranking** | **LightGBM LambdaRank, 22 features** | **0.1348** | **0.0184** | **0.1951** | 0.0282 |

(Notebook results, single seed. Pipeline re-runs with 3 seeds: NDCG@10 **0.0202 ± 0.0008**, Recall@100
**0.1347 ± 0.0027**; see §5.)

**Findings**
- A recency heuristic beat EASE ~2x. Time drift and new items dominate this data.
- The text tower produced the first non-zero recall on new games. An item-age feature *hurt* the tower
  (in-batch negatives' age was computed at their own timestamps → distribution shift); recency was moved
  into the ranker instead, where it helped.
- Two candidate sources (two-tower top-200 + recent-pop top-100) raise the candidate ceiling to 0.1664;
  the ranker gives +38% Recall@100 and +26% NDCG@10 over the two-tower order.
- The ranker under-serves new games (cold R@100 0.0378 → 0.0282): it was trained on a val window with 27%
  cold share vs 46% at test.

**ANN (25,612 items):** Flat 0.36 ms p95 / HNSW 0.38 / IVF 0.13 / IVF-PQ 0.14. HNSW at 98.6% index recall
left end-to-end Recall@100 unchanged (0.0976 vs 0.0975). Index recall is a proxy; Flat is chosen at this scale.

**Beyond accuracy (top-10):**

| | Coverage@10 | New-item share | Cold recall@10 | NDCG@10 |
|---|---|---|---|---|
| Ranker | 11.1% | 2.4% | 0.0055 | 0.0184 |
| Ranker + **2 new-item slots** | 12.5% | **14.2%** | **0.0141** | 0.0186 |

One re-rank rule fixed the ranker's learned bias (new-item exposure ~6x, cold recall 2.6x) at no NDCG cost.
Users without history get "popular in the last 30 days" (chosen on val).

## 3. Serving

`FastAPI` → user tower (last 20 items) → FAISS top-200 + recent-pop top-100 → 22 features → LightGBM →
new-item slots → top-10. User state in Redis (or in-memory); one shared code path for training and serving.

| Measurement (real data, laptop CPU) | Result |
|---|---|
| Single request | 7.7 ms (user tower 2.0, ANN 2.3, features 1.4, ranker 1.6) |
| 1 worker, 20 concurrent users | ~430 RPS, p95 32 ms; CPU ~20% → GIL-bound → multi-worker + Redis + load lock |
| Training/serving parity | features: max diff 0 · end-to-end top-k: 300/300 users identical (near-ties explained) |

## 4. Training pipeline and evaluation gate

`python -m pipeline.run`: prepare → train_retrieval → build_index → train_ranker → evaluate → gate → promote.
Every run is a versioned folder; MLflow (local SQLite) tracks params, metrics, tags, artifacts; production is
a pointer file (`production.json`, atomic swap) with an append-only history for rollback.

- **Reproducible:** the pipeline reproduces the notebook numbers exactly (0.0184 / 0.1348 / 0.0975 / 0.0916).
- **GPU:** two-tower ~3 s/epoch; a full run takes ~5 min, of which evaluation is ~75%.

**Gate (decides promotion):** sanity → beats recent popularity → candidate data not older than production's
(leakage guard) → **paired bootstrap regression test**: production is re-evaluated on the candidate's test
window, per-user differences are bootstrapped (2,000×), block if the 95% CI is entirely below −1% or the
point drop exceeds 5%. Metrics: ranker NDCG@10, Recall@100, and candidate recall (so retrieval cannot
silently degrade behind the ranker).

Measured CI half-widths on real data (21,653 users): **NDCG@10 ±5.2%**, Recall@100 ±1.7%, candidate
recall ±1.5%. Two rules were tried and rejected before this one:
- a fixed 1% tolerance on point estimates ignores evaluation noise entirely;
- non-inferiority ("prove it is not >2% worse") blocks even an identical model, because sparse NDCG@10 has
  a CI wider than the margin.

## 5. Stability studies

**Seeds (full pipeline, GPU, 3 seeds):** NDCG@10 0.0202 ± 0.0008 (~4%), Recall@100 0.1347 ± 0.0027.

**Ranker early stopping:** a single 15% validation split is ~1,150 users. On real data it stopped at 14 and
18 trees in two runs. Fix: pick the number of trees from the mean validation curve of 3 user folds, then
refit on all users. Ranker study (retrieval fixed, 5 seeds):

| | Trees chosen | NDCG@10 | Recall@100 |
|---|---|---|---|
| Single split | 32–66 | 0.0206 ± 0.0008 | 0.1355 ± 0.0012 |
| 3-fold mean curve | 47–110 | 0.0210 ± 0.0008 | 0.1363 ± 0.0010 |

Individual folds chose anywhere from 20 to 148 trees, and the mean curve absorbed that noise. NDCG is
better on 3/5 seeds and tied on 2/5, so the mean gain is small and not established. Seed variance is
unchanged: it comes from bagging and user sampling, not from the stopping point. Kept as a robustness fix.

## 6. Time-travel replay: how fast do models go stale?

`python -m pipeline.replay` runs the full pipeline (including the gate) at 6 quarterly cutoffs (2022-01 →
2023-03), in an isolated namespace, then evaluates every model on every later window.

| Model age | NDCG@10 vs fresh model (same window) | Recall@100 vs fresh |
|---|---|---|
| 90 days | −11.5% | −7.7% |
| 180 days | −11.1% | −12.9% |
| 270 days | −29.6% | −17.3% |
| 360 days | −23.5% | −21.1% |

Fresh vs 90-day-old model in the gate: +27% [+14%, +42%], +13.7% [+1.8%, +26%] NDCG@10; +16.9% [+8%, +27%]
Recall@100 (95% paired CIs). Recall decays monotonically; NDCG is noisier (1–5 pairs per cell).

| Policy (6 windows) | Mean NDCG@10 | vs always retrain |
|---|---|---|
| Always retrain (quarterly) | 0.0303 | — |
| Gated (what the pipeline did) | 0.0307 | +1.0% |
| Every 2 quarters | 0.0277 | −8.8% |
| Never retrain | 0.0263 | −13.5% |

The gate blocked one retrain (2022-06-30, whose ranker stopped at 14 trees) and kept the previous model.
That is why "gated" is not below "always". Conclusion: a retrain costs ~3–4 GPU-minutes and quarterly
staleness already costs ~10%, so retrain at least monthly. Whether monthly beats quarterly is measurable
with `--every-days 30`; not yet run.

## 7. Zero-downtime deployment, monitoring, orchestration

- **Hot swap:** each API worker polls `production.json`, loads the new model in a background thread,
  smoke-tests it with real requests, then swaps one reference. The user store is not reloaded. *Synthetic
  data:* 8,689 requests across 2 swaps, 0 errors, max 206 ms; a corrupt model was rejected (old model kept
  serving, 0 errors); Redis + 2 workers, both swapped within ~2 s.
- **Monitoring:** `/metrics` (multi-process aware): traffic by arm/source, latency by arm and stage, errors,
  fallback share, top-score drift, new-item share, reloads, model per worker. 12-panel Grafana dashboard
  and 6 alert rules. Validated with a real Prometheus 2.53 scraping the API (`promtool` clean, all 14
  dashboard queries returned data).
- **Orchestration:** an Airflow DAG with per-step retries. The gate is a branch: a failed gate makes promote
  *skipped*, and the DAG stays green. A `verify_rollout` task waits until the API serves the new run and
  rolls back otherwise. Backfill with `time_travel=true` = replay. Both paths were tested with
  `airflow dags test` (Airflow 2.10.5). A Windows Task Scheduler script covers the single-machine case.
- **CI:** GitHub Actions runs pytest plus an end-to-end smoke test on generated synthetic data: pipeline,
  a deliberately bad model the gate must block, API, parity, A/B, metrics.

## 8. A/B testing framework (simulated users)

Salted-hash assignment by user. Server-side exposure log and client click log joined by `request_id`.
Analysis: data checks → SRM (χ²) → A/A calibration → per-user lift with delta-method CI → CUPED →
guardrails (latency, errors, fallback, coverage) → power/MDE → decision. Also a peeking demo and
team-draft interleaving.

There are no real users. Traffic is simulated by replaying each user's real future interactions through a
position-biased click model against the live API. **The framework is real; effect sizes are simulated.**

*Synthetic-data results:* A/A false-positive rate 5.6% (target 5%); peeking 20 times inflated false
positives from ~5% to 22–28%. Offline paired CI for the same two models was ~±2%, versus ±13% for the
online A/B: the offline gate is the cheap, sensitive filter, and A/B answers what offline cannot.

**Real-data A/B: policy test, 2 new-item slots (control) vs none.** All 91,381 users with history, one
simulated visit each, 0 failed visits.

| Check | Result |
|---|---|
| Data | 0 orphan clicks, 0 users in two arms |
| SRM | 45,956 vs 45,425, χ² p = 0.079 → OK |
| A/A false-positive rate | 4.0% (500 splits) |
| Clicks per user (no slots vs slots) | −0.79%, 95% CI [−6.9%, +5.4%], p = 0.80 |
| Power | MDE 8.8% relative; a 2% effect would need ~886k users per arm |
| CUPED variance reduction | 0.0% (see below) |
| Guardrail: new-item share | 12.4% → **2.6%** (−79%) → FAIL |
| Guardrail: catalog coverage | 3,763 → 3,528 distinct items on equal samples (−6%) → FAIL |
| Interleaving (same 91k users) | +0.008, p = 0.62 → no preference |
| Peeking (A/A, 20 looks) | 23.6% false positives vs 5.0% with one look |

**Decision: keep the slots (treatment = DON'T SHIP).** Removing them did not move clicks measurably. The CI
bounds any click gain from removal at about +5%. Meanwhile it cut new-item exposure ~5x and catalog coverage
by 6%, which matches the offline result (14.2% vs 2.4% new-item share).

**What the first attempt taught.** On a 5,111-user subsample the same experiment showed "+35% clicks,
p = 0.02", and the old analyzer said SHIP. That subsample had an MDE of 39.5%. With all users, the effect is
−0.8%. This is a textbook type-M error: an underpowered test that reaches significance overstates the effect.
The analyzer now refuses to decide when MDE > 10%.

**CUPED reduced no variance**, on real or synthetic data. The covariate (pre-period review counts) does not
predict whether a user returns and clicks during the test window. CUPED needs the same metric measured in a
pre-period (past clicks), which simulated users do not have.

## 9. Incidents and what they changed

| Incident | Cause | Fix |
|---|---|---|
| Replay overwrote the real production pointer | `config.py` with `namespace` support was never shipped; the override was silently ignored | Recovery script (moves runs, restores pointer, nothing deleted). Replay now refuses to run outside a replay namespace |
| Gate let an older-data model replace a newer one | Production was re-scored on a window it had trained on | Leakage guard: the candidate's test window must start ≥ production's |
| Gate decisions were noise-driven | 1% point tolerance vs ±5% evaluation noise | Paired bootstrap regression test |
| Continuing an old run used its old gate policy | The saved config included deployment policy | Gate and MLflow policy always come from the current config |
| Parity check failed on real data | LightGBM split thresholds amplified 1e-7 float differences | Two-level parity: exact features + explained near-ties end to end |
| 1 worker plateaued at ~430 RPS at 20% CPU | GIL | Multi-worker + Redis + a load lock so only one worker bootstraps |
| First real-data A/B lost 74% of visits | Simulator opened a new TCP connection per request → Windows ephemeral-port exhaustion (WinError 10048) | Keep-alive connection per thread + retries; warn if >1% of visits fail |
| First real-data A/B said "SHIP" (+35% clicks, p = 0.02) | 5,111 users, MDE 39.5%: an underpowered "significant" result overstates the effect (type-M); interleaving on the same users found nothing (p = 0.72); coverage dropped 9% with no guardrail | Decision now requires MDE ≤ 10% (else "no decision" + users needed); coverage and new-item-share guardrails added |

## 10. Limitations

- Reviews are a proxy for purchases. The 5-core keeps only the 3.4% most active users, and the k-core is
  computed on the full timeline (mild leakage, kept for benchmark comparability).
- The item catalog is a 2023 snapshot: retrieval can surface games before their release date (no
  release-date filter yet).
- One category (L1, 25k items). ANN trade-offs at 500k+ items were only shown on synthetic vectors.
- A/B effects are simulated (one visit per user, no novelty or counterfactual interest). Interleaving
  showed no sensitivity gain here, because within-user click noise dominates. CUPED had no pre-period
  metric to work with.
- Not yet run on the laptop: the Docker monitoring stack and GitHub CI.

## Reproduce

```bash
pip install -r requirements.txt
python -m pipeline.run                                      # train → gate → promote
uvicorn serve.app:app --port 8000                           # serve production
python -m serve.parity_check --n 300
python -m pipeline.replay --name q90 --start 2022-01-01 --end 2023-04-01 --every-days 90
python -m ab.simulate --api http://127.0.0.1:8000 --n-users 20000 && python -m ab.analyze
bash scripts/ci_smoke.sh                                    # everything on synthetic data, ~1 min
```
