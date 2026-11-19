#!/usr/bin/env bash
# P2 phase D, end to end on a real S3 server (CI: the SeaweedFS container; laptop: make s3-up) with synthetic data:
#   pipeline (MODEL_REGISTRY=s3) publishes + promotes -> engine serves FROM S3 -> second run promoted -> the engine's
#   watcher swaps by itself -> admin rollback moves the S3 pointer back -> the engine follows.
# Needs S3_ENDPOINT_URL, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY (and S3_BUCKET). Everything goes under a throwaway
# namespace (deleted at the end), so it never touches real runs even on the laptop's server.
set -euo pipefail
: "${S3_ENDPOINT_URL:?set S3_ENDPOINT_URL (and the S3_* keys) to a running S3 server}"
W=${CI_WORK:-$(mktemp -d)}
NS=ci/registry-$(python -c "import uuid; print(uuid.uuid4().hex[:8])")
PORT=${REGISTRY_API_PORT:-8097}
export RANKFORGE_DATA=$W/data RANKFORGE_REPO=$W/repo RANKFORGE_CATEGORY=Video_Games
export MLFLOW_DISABLE_AGENT_HINT=1 EXPERIMENT_CONFIG=$W/no-experiment.yaml
export MODEL_REGISTRY=s3 MODEL_NAMESPACE=$NS MODEL_CACHE_DIR=$W/engine-cache MODEL_WATCH_SECONDS=2
mkdir -p "$RANKFORGE_REPO"
API=""
cleanup() {
  [ -n "$API" ] && kill "$API" 2>/dev/null || true
  python - "$NS" <<'PY' || true
import sys
from pipeline.storage import Registry, S3Config
reg = Registry(S3Config.from_env(), "Video_Games", sys.argv[1]); n = 0
for prefix in (reg.models + "/", reg.gold + "/"):
    for o in reg.s3.list_objects_v2(Bucket=reg.cfg.bucket, Prefix=prefix).get("Contents", []):
        reg.s3.delete_object(Bucket=reg.cfg.bucket, Key=o["Key"]); n += 1
print(f"namespace {sys.argv[1]}: {n} objects deleted")
PY
}
trap cleanup EXIT
health() { curl -sf "127.0.0.1:$PORT/health"; }
field() { health | python -c "import json,sys; print(json.load(sys.stdin)[sys.argv[1]])" "$1"; }

echo "== 1. synthetic gold ($W, namespace $NS)"
python scripts/make_synthetic_gold.py --out "$RANKFORGE_DATA" --users 2500 --items 1000
FAST="--set two_tower.text_encoder=tfidf --set two_tower.epochs=2 --set ranker.n_estimators=60 --set ranker.max_train_users=3000 --set mlflow.enabled=false --set namespace=$NS"

echo "== 2. pipeline in S3 mode: publish + promote reg-a"
python -m pipeline.run --run-id reg-a $FAST
test ! -f "$RANKFORGE_REPO/models/Video_Games/$NS/production.json" || { echo "S3 mode wrote a LOCAL pointer"; exit 1; }

echo "== 3. engine serves from S3"
uvicorn serve.app:app --host 127.0.0.1 --port "$PORT" > "$W/api.log" 2>&1 &
API=$!
for i in $(seq 120); do health > /dev/null && break; sleep 1; done
health | python -c "
import json, sys
h = json.load(sys.stdin)
assert h['model_source'] == 's3' and h['model_run'] == 'reg-a' and h['production_version'] == 1, h
assert 'engine-cache' in h['model_dir'], h['model_dir']
print('engine serves', h['model_run'], 'v1 from', h['model_dir'])"
curl -sf "127.0.0.1:$PORT/recommend/u-ci?k=5" > /dev/null

echo "== 4. second run promoted -> the engine's watcher swaps by itself"
python -m pipeline.run --run-id reg-b $FAST --force
for i in $(seq 120); do [ "$(field model_run)" = "reg-b" ] && break; sleep 1; done
[ "$(field model_run)" = "reg-b" ] && [ "$(field production_version)" = "2" ] || { tail -30 "$W/api.log"; exit 1; }
echo "swapped to reg-b v2"

echo "== 5. admin rollback moves the S3 pointer and the engine follows"
curl -sf -X POST "127.0.0.1:$PORT/admin/rollback" | python -c "import json,sys; r=json.load(sys.stdin); assert r['rolled_back_to']=='reg-a', r; print('rollback ->', r['rolled_back_to'], r['swap']['status'])"
[ "$(field model_run)" = "reg-a" ] && [ "$(field production_version)" = "3" ] || { tail -30 "$W/api.log"; exit 1; }
python - "$NS" <<'PY'
import sys
from pipeline.pointer import S3Pointer
from pipeline.storage import Registry, S3Config
h = S3Pointer(Registry(S3Config.from_env(), "Video_Games", sys.argv[1])).history()
assert [(x["version"], x["run_id"]) for x in h] == [(1, "reg-a"), (2, "reg-b"), (3, "reg-a")], h
print("S3 history:", [(x["version"], x["run_id"], x["reason"]) for x in h])
PY
echo "REGISTRY E2E PASSED"
