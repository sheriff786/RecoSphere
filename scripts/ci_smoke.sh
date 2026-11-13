#!/usr/bin/env bash
# End-to-end smoke test on synthetic data (Phase 7g). Same script runs in GitHub Actions and locally:
#   bash scripts/ci_smoke.sh
# pipeline (train -> index -> ranker -> evaluate -> gate -> promote) -> API -> parity -> A/B -> metrics
set -euo pipefail
W=${CI_WORK:-$(mktemp -d)}
export RANKFORGE_DATA=$W/data RANKFORGE_REPO=$W/repo RANKFORGE_CATEGORY=Video_Games
export MLFLOW_DISABLE_AGENT_HINT=1 MODEL_WATCH_SECONDS=0 EXPOSURE_DIR=$W/ab_logs EXPERIMENT_CONFIG=$W/active.yaml
mkdir -p $RANKFORGE_REPO
echo "== 1. synthetic gold"; python scripts/make_synthetic_gold.py --out $RANKFORGE_DATA --users 2500 --items 1000
FAST="--set two_tower.text_encoder=tfidf --set two_tower.epochs=2 --set ranker.n_estimators=60 --set ranker.max_train_users=3000 --set mlflow.enabled=false"
echo "== 2. pipeline"; python -m pipeline.run --run-id ci $FAST
test -f $RANKFORGE_REPO/models/Video_Games/production.json
echo "== 3. bad run must be blocked by the gate (exit 2)"
set +e; python -m pipeline.run --run-id ci-bad $FAST --set ranker.n_estimators=1 --set two_tower.epochs=1 --set ranker.n_tt=5 --set ranker.n_pop=5; code=$?; set -e
[ "$code" = "2" ] || { echo "expected gate failure (exit 2), got $code"; exit 1; }
echo "== 4. API + experiment (policy test: new-item slots 2 vs 0)"
cat > $EXPERIMENT_CONFIG <<YAML
name: ci-policy
salt: ci-policy-v1
arms:
  - {name: control, model: production, traffic: 0.5}
  - {name: no_slots, model: production, traffic: 0.5, new_item_slots: 0}
YAML
uvicorn serve.app:app --host 127.0.0.1 --port 8099 > $W/api.log 2>&1 &
API=$!; trap 'kill $API 2>/dev/null || true' EXIT
for i in $(seq 120); do curl -sf 127.0.0.1:8099/health > /dev/null && break; sleep 1; done
curl -sf 127.0.0.1:8099/health | python -c "import sys,json; d=json.load(sys.stdin); assert d['model_run']=='ci', d; assert d['experiment']['name']=='ci-policy', d; print('health ok', d['model_run'])"
echo "== 5. training/serving parity"; python -m serve.parity_check --n 100
echo "== 6. A/B simulation + analysis"
python -m ab.simulate --api http://127.0.0.1:8099 --n-users 1500 --threads 4
python -m ab.analyze --logs $EXPOSURE_DIR --experiment $EXPERIMENT_CONFIG --out $W/ab_report > $W/ab_report.txt; sed -n 5,8p $W/ab_report.txt
python -c "import json,sys; r=json.load(open('$W/ab_report/report.json')); assert not r['srm']['srm'] and r['data']['orphan_clicks']==0 and r['data']['users_in_multiple_arms']==0, r['data']; print('A/B data checks ok')"
echo "== 7. metrics"; curl -sf 127.0.0.1:8099/metrics | grep -q 'recosphere_requests_total{arm="no_slots"' && echo "metrics ok"
echo "SMOKE TEST PASSED"
