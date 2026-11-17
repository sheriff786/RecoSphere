#!/usr/bin/env bash
# Browser end-to-end test on a disposable synthetic stack. Same script in GitHub Actions and locally:
#   E2E_PG_URL=<postgres server url> make e2e-synthetic          (or: bash scripts/e2e_stack.sh)
# synthetic gold -> 2 promoted runs (so Roll back has a target) -> engine :8098
# -> a THROWAWAY Postgres database (scripts/e2e_db.py): Alembic + synthetic catalog + admin -> app_api :8198
# -> app (standalone build, like the Docker image) :3098 -> Playwright with E2E_ALLOW_ROLLBACK=1.
# Nothing touches the real models, data or app database: files live in $W, the database is dropped at the end
# (also on failure). E2E_PG_URL: CI = its postgres service; laptop = Neon's DIRECT url (only CREATE/DROP DATABASE
# run there). Ports differ from `make up` so both can run at once.
# Needs: python with requirements-ci, web/node_modules (npm ci), Chromium (npx playwright install chromium).
set -euo pipefail
: "${E2E_PG_URL:?set E2E_PG_URL to a Postgres server url (any database on it) for the throwaway e2e database}"
W=${E2E_WORK:-$(mktemp -d)}
API_PORT=${E2E_API_PORT:-8098}
APP_API_PORT=${E2E_APP_API_PORT:-8198}
WEB_PORT=${E2E_WEB_PORT:-3098}
export RANKFORGE_DATA=$W/data RANKFORGE_REPO=$W/repo RANKFORGE_CATEGORY=Video_Games
export MLFLOW_DISABLE_AGENT_HINT=1 MODEL_WATCH_SECONDS=0
export CUDA_VISIBLE_DEVICES=-1    # CPU like CI; a laptop GPU hiccup must not fail the e2e ("" is unset on Windows)
export EXPERIMENT_CONFIG=$W/no-experiment.yaml          # missing file = no A/B; never the repo's real active.yaml
mkdir -p "$RANKFORGE_REPO"
PIDS=()
DB_URL=""
cleanup() {
  for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done
  if [ -n "$DB_URL" ]; then python scripts/e2e_db.py drop "$E2E_PG_URL" "$DB_URL" && echo "e2e database dropped"; fi
}
trap cleanup EXIT

echo "== 1. synthetic gold + two promoted runs ($W)"
python scripts/make_synthetic_gold.py --out "$RANKFORGE_DATA" --users 2500 --items 1000
FAST="--set two_tower.text_encoder=tfidf --set two_tower.epochs=2 --set ranker.n_estimators=60 --set ranker.max_train_users=3000 --set mlflow.enabled=false"
python -m pipeline.run --run-id e2e-a $FAST
python -m pipeline.run --run-id e2e-b $FAST
python -c "import json,sys; p=json.load(open(sys.argv[1])); assert p['run_id']=='e2e-b', p" \
  "$RANKFORGE_REPO/models/Video_Games/production.json"

echo "== 2. engine :$API_PORT"
uvicorn serve.app:app --host 127.0.0.1 --port "$API_PORT" > "$W/api.log" 2>&1 &
PIDS+=($!)

echo "== 3. throwaway app database + app_api :$APP_API_PORT"
DB_URL=$(python scripts/e2e_db.py create "$E2E_PG_URL")
export DATABASE_URL=$DB_URL DATABASE_URL_DIRECT=$DB_URL SERVICE_TOKEN=e2e-service-token-0123456789
python -m alembic upgrade head
python scripts/load_catalog.py --data "$RANKFORGE_DATA" | tail -3
ADMIN_PASSWORD=admin12345 python -m app_api.cli create-admin --email admin@recosphere.dev
uvicorn app_api.main:app --host 127.0.0.1 --port "$APP_API_PORT" > "$W/app_api.log" 2>&1 &
PIDS+=($!)

# HOSTNAME=0.0.0.0 like the Docker image. Bound to 127.0.0.1, Next rewrites 127.0.0.1 to "localhost" in
# redirects, so the browser (on 127.0.0.1) would land on another origin: a test-setup artefact, not an app bug.
echo "== 4. app :$WEB_PORT (standalone build)"
(
  cd web
  npx next build > "$W/web-build.log" 2>&1 || { tail -40 "$W/web-build.log"; exit 1; }
  rm -rf .next/standalone/.next/static && cp -r .next/static .next/standalone/.next/static
  mkdir -p public && rm -rf .next/standalone/public && cp -r public .next/standalone/public
)
PORT=$WEB_PORT HOSTNAME=0.0.0.0 ENGINE_URL=http://127.0.0.1:$API_PORT \
  APP_API_URL=http://127.0.0.1:$APP_API_PORT APP_API_TOKEN=$SERVICE_TOKEN \
  AUTH_SECRET=e2e-only-secret-0123456789abcdef0123456789 COOKIE_SECURE=0 \
  node web/.next/standalone/server.js > "$W/web.log" 2>&1 &
PIDS+=($!)

echo "== 5. playwright"
set +e
(cd web && E2E_BASE_URL=http://127.0.0.1:$WEB_PORT E2E_ENGINE_URL=http://127.0.0.1:$API_PORT \
  E2E_ALLOW_ROLLBACK=1 npx playwright test)
code=$?
set -e
if [ $code -ne 0 ]; then
  echo "---- engine log (tail)"; tail -30 "$W/api.log"; echo "---- app_api log (tail)"; tail -30 "$W/app_api.log"
  echo "---- app log (tail)"; tail -30 "$W/web.log"
  exit $code
fi
python -c "import json,sys; p=json.load(open(sys.argv[1])); assert p['run_id']=='e2e-a', p; print('rollback landed on', p['run_id'])" \
  "$RANKFORGE_REPO/models/Video_Games/production.json"
echo "E2E PASSED"
