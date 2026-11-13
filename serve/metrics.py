"""
Prometheus metrics (Phase 7g). Scraped from GET /metrics.

Multi-worker: uvicorn --workers N = N processes, each with its own counters. Set PROMETHEUS_MULTIPROC_DIR
(an empty folder, wiped at container start) and /metrics aggregates all workers. Without it, /metrics
shows only the worker that answered the scrape (fine for 1 worker / dev).

What we watch and why:
  requests_total{source,arm,model_run}   traffic, fallback share (cold users / errors), A/B split sanity
  request_seconds{arm}                   latency SLO (p95 < 50 ms) per arm - a treatment can be slower
  stage_seconds{stage}                   which stage got slow (user tower, ANN, features, ranker)
  errors_total{kind}                     recommend failures (served a popular list instead), bad events
  top_score{model_run}                   score drift: a broken/new model shifts this distribution
  new_item_share{arm}                    is the new-item re-rank rule still doing its job
  feedback_total{arm,action}             online clicks by arm (live A/B monitor; analysis is offline)
  model_reloads_total{result}            hot-swap health
  model_info{model_run,emb_version}      which model each worker serves (1 = live)
"""
import os

from prometheus_client import (CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest,
                               multiprocess)

MULTI = bool(os.environ.get("PROMETHEUS_MULTIPROC_DIR"))
LAT_BUCKETS = (.002, .005, .0075, .01, .015, .02, .03, .05, .075, .1, .2, .5, 1, 2)

REQUESTS = Counter("recosphere_requests_total", "recommend requests", ["source", "arm", "model_run"])
LATENCY = Histogram("recosphere_request_seconds", "end-to-end recommend latency", ["arm"], buckets=LAT_BUCKETS)
STAGE = Histogram("recosphere_stage_seconds", "per-stage latency", ["stage"], buckets=LAT_BUCKETS)
ERRORS = Counter("recosphere_errors_total", "errors", ["kind"])
TOP_SCORE = Histogram("recosphere_top_score", "ranker score of the #1 item", ["model_run"],
                      buckets=(-4, -2, -1, -.5, 0, .5, 1, 2, 4, 8))
NEW_SHARE = Histogram("recosphere_new_item_share", "share of new items in the served list", ["arm"],
                      buckets=(0, .1, .2, .3, .5, 1))
FEEDBACK = Counter("recosphere_feedback_total", "client feedback", ["arm", "action"])
EVENTS = Counter("recosphere_events_total", "POST /event", ["recorded"])
RELOADS = Counter("recosphere_model_reloads_total", "model hot-swaps", ["result"])
MODEL_INFO = Gauge("recosphere_model_info", "model served by this worker", ["model_run", "emb_version"],
                   multiprocess_mode="max")


for _k in ("recommend", "unknown_item"):                      # export 0 instead of "no data"
    ERRORS.labels(_k)
for _r in ("swapped", "failed"):
    RELOADS.labels(_r)


def observe_recommend(out: dict, arm: str, model_run: str, seconds: float):
    REQUESTS.labels(out["source"], arm, model_run).inc()
    LATENCY.labels(arm).observe(seconds)
    for stage, ms in (out.get("timings_ms") or {}).items():
        if stage != "total":
            STAGE.labels(stage).observe(ms / 1e3)
    items = out["items"]
    if items:
        if items[0]["score"] is not None:
            TOP_SCORE.labels(model_run).observe(items[0]["score"])
        NEW_SHARE.labels(arm).observe(sum(x["is_new"] for x in items) / len(items))


def set_model(old_run, old_emb, new_run, new_emb):
    if old_run is not None:
        MODEL_INFO.labels(old_run, old_emb).set(0)
    MODEL_INFO.labels(new_run, new_emb).set(1)


def render() -> tuple[bytes, str]:
    if MULTI:
        reg = CollectorRegistry()
        multiprocess.MultiProcessCollector(reg)
        return generate_latest(reg), CONTENT_TYPE_LATEST
    return generate_latest(), CONTENT_TYPE_LATEST
