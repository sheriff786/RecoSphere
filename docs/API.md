# RecoSphere API reference

Every HTTP endpoint in the system: what it does, how to call it, and which code does the work.

**Keep this file in sync.** Any change that adds, changes or removes an endpoint updates this file in the
same commit. `tests/test_api_doc.py` fails CI when an engine route or a web `/api/*` route is missing here.

```
browser ──► web app (Next.js, :3000)  /api/*  ── server-side fetch ──►  engine (FastAPI, :8000)
            pages render on the server             lib/engine.ts            Swagger: /docs, /redoc
```

- **The browser never calls the engine.** It talks to the web app only. The web app calls the engine from
  the server (`web/lib/engine.ts`), so the engine needs no CORS and its admin token never reaches a browser.
- **Engine user ids:** app accounts are `app:<id>`; "real players" are dataset user ids (Amazon reviewer ids).
- **Engine clock:** frozen at the start of the test window (`t_ref` in `/health`, July 2021 on the real data).
  Events without `ts` are stamped with it. Real time comes in P4.
- **Live, interactive docs:** `http://127.0.0.1:8000/docs` (Swagger UI), `/redoc`, `/openapi.json`.

## 1. Engine API (`serve/`, FastAPI)

`serve/app.py` creates the app and wires the routers; shared state is in `serve/deps.py`
(`STATE["mgr"]` = the `ModelManager`, `STATE["exposures"]` / `STATE["clicks"]` = JSONL loggers).

| Method | Path | Purpose | Router |
|---|---|---|---|
| GET | `/recommend/{user_id}` | top-k recommendations (A/B routed, exposure-logged) | `routers/recommend.py` |
| GET | `/explain/{user_id}` | why items are in the user's feed (SHAP reasons) | `routers/recommend.py` |
| POST | `/event` | record an interaction (play) → user history | `routers/events.py` |
| POST | `/feedback` | record a click/play on a served list (A/B outcome) | `routers/events.py` |
| GET | `/catalog/items` | game cards by id | `routers/catalog.py` |
| GET | `/catalog/popular` | trending games (optionally only new ones) | `routers/catalog.py` |
| GET | `/catalog/search` | title search | `routers/catalog.py` |
| GET | `/catalog/similar/{item_id}` | nearest games in embedding space | `routers/catalog.py` |
| GET | `/users/demo` | real dataset players to explore as | `routers/catalog.py` |
| GET | `/users/{user_id}/history` | a user's recent games | `routers/catalog.py` |
| GET | `/health` | model, store, experiment and reload status | `routers/admin.py` |
| GET | `/metrics` | Prometheus metrics | `routers/admin.py` |
| POST | `/admin/reload` | reload production model + experiment config now | `routers/admin.py` |
| GET | `/admin/summary` | health + live counters + production history | `routers/admin.py` |
| POST | `/admin/rollback` | point production at the previous model, hot-swap | `routers/admin.py` |

**Admin auth:** if the engine runs with `ADMIN_TOKEN`, the three `/admin/*` routes need the header
`X-Admin-Token: <token>`, else `403 {"detail": "bad admin token"}`. **With no `ADMIN_TOKEN` (the default
in `.env.example`) they are open to anyone who can reach the engine port.** Locking this down is P7.

**Errors:** FastAPI format `{"detail": ...}`. Query validation failures (e.g. `k` out of range) are `422`.

### GET `/recommend/{user_id}`
The main endpoint: the top-k games for a user.

| Query | Default | Meaning |
|---|---|---|
| `k` | 10 | list length, 1-50 |
| `rerank` | true | apply the new-item slots re-rank (see below) |
| `debug` | false | include `timings_ms` per stage |
| `force_arm` | – | serve a specific A/B arm (QA only; logged with `forced=true`, excluded from analysis); unknown arm → `400` |
| `meta` | false | return full game cards (as `/catalog/items`) instead of id/title/is_new |

**How it works** (`serve/recommender.py: Recommender.recommend`):
1. **A/B routing** (`serve/experiments.py: ExperimentManager.route`): salted sha256 of the user id picks the
   arm of the active experiment (`experiments/active.yaml`); each arm can be another model or policy.
2. **User state** from the feature store (`serve/feature_store.py`, Redis or in-memory): history, seen items.
3. **No history → `source: "fallback"`**: the popularity fallback list.
4. **Candidates** (`score_candidates`): user tower embeds the history (`retrieval/two_tower.py`) → FAISS top
   `n_tt` + recent-popularity top `n_pop`, seen items removed.
5. **Features + ranking**: `ranking/pipeline.py: assemble` builds the 22 features (the same code as training:
   parity), LightGBM LambdaRank scores them, ties broken by item id.
6. **Re-rank**: `new_item_slots` makes sure up to `NEW_ITEM_SLOTS` (default 2) new games are in the top-k,
   taken from the ranker's top 100. If none of those 100 is new, fewer appear.
7. **Logging**: Prometheus counters/histograms (`serve/metrics.py`); with `EXPOSURE_DIR` set, one JSON line
   per request in `exposures-<pid>.jsonl` (request id, arm, items, latency). `ab/analyze.py` joins these with clicks.
8. **Never fails the page**: an exception logs, counts `recosphere_errors_total{kind="recommend"}` and returns
   the fallback list with `source: "fallback_error"`.

Response (real model; `k=2&debug=true`, titles shortened):
```json
{
  "user_id": "AEPQB547GDARJWCYCDJO4X2WH5CQ",
  "source": "personalized",
  "items": [
    {"item_id": "B097PBVHYP", "title": "Xbox PC Wireless Controller – Space Jam: ...", "is_new": true, "score": 0.31845},
    {"item_id": "B0BPMJ8TRV", "title": "PowerA Spectra Infinity Enhanced Wired Controller ...", "is_new": true, "score": 0.17372}
  ],
  "timings_ms": {"store_get": 0.0, "user_tower": 0.0, "ann_candidates": 0.0, "features": 0.0, "ranker": 0.0, "rerank": 0.0, "total": 0.0},
  "request_id": "c58cf3a86cce4a22bad3688d10dd6c11",
  "experiment": "new-slots-2-vs-0",
  "arm": "control",
  "model_run": "20261111-171751"
}
```
(`timings_ms` values are placeholders here; they are real per request.) `source` is `personalized`,
`fallback` or `fallback_error`. `experiment` is `null` and `arm` is `control` when no experiment is active.

### GET `/explain/{user_id}`
Why games are in this user's feed. Used by the app's "Why this?" panel.

| Query | Default | Meaning |
|---|---|---|
| `items` | the user's top 10 | comma-separated item ids to explain |

**How it works** (`serve/product.py: Product.explain`), with the model of the user's A/B arm:
re-scores the user's candidates, then for each item takes the LightGBM feature contributions
(SHAP, `booster.predict(pred_contrib=True)`), maps positive ones to plain-language reasons (`REASON` table;
user-only features are skipped because they do not separate items), keeps the top 3 groups, and finds the
closest game in the user's history by item-embedding cosine ("because you played").

Response: `{"user_id", "arm", "model_run", "source", "items": {<item_id>: {...}}}`. Per item:
`in_candidates`, `rank`, `n_candidates`, `score`, `sources` (`taste model` / `trending`),
`because_you_played`, `reasons: [{group, text, weight, feature}]`. An item outside the candidate set gets
`in_candidates: false` and a "Similar to ..." reason. A user without history gets `source: "fallback"` and a
generic "popular" reason.

### POST `/event`
Records that a user interacted with a game (the app sends one per onboarding pick and per Play). The next
`/recommend` already uses it: the user tower sees the new history and the game is excluded as seen.

Body: `{"user_id": str, "item_id": str, "ts": int | null, "rating": float | null}`. `ts` defaults to the
engine clock `t_ref`, `rating` to 5.0.

**How it works:** `Recommender.add_event` → `feature_store.add_event`: an unknown user id gets a new
internal `uid`; the game is appended to the history (last `hist_len` kept), the seen list, the time span,
rating sum and publisher counts. Counted in `recosphere_events_total{recorded}`.

Response: `{"user_id", "uid", "item_id", "recorded": bool}`. `recorded: false` = the user has already seen
this game; only the first interaction counts (dedup rule from notebook 02), nothing changes.
Unknown item → `404 {"detail": "unknown item_id <id>"}`.

### POST `/feedback`
Client-side outcome of a served list (the A/B metric).

Body: `{"request_id": str, "user_id": str, "item_id": str, "action": str = "click"}` (the app sends `click`
or `play`).

**How it works:** the arm is looked up again from the user id (assignment is deterministic, so a client
cannot claim a wrong arm), counted in `recosphere_feedback_total{arm, action}`, and with `EXPOSURE_DIR` set
written to `clicks-<pid>.jsonl`. Response: `{"ok": true}`.

### GET `/catalog/items`
Game cards by id. Query: `ids` (required, comma-separated). Unknown ids are skipped.

**How it works:** `Product.items` → `_meta`: title, `platforms` and `categories` parsed from the item text,
release `year`, `description` (max 600 chars), `is_new`, `plays_90d`, `trending_rank`.

Response: `{"items": [card, ...]}`. Card fields: `item_id, title, platforms, categories, year, description,
is_new, plays_90d, trending_rank`.

### GET `/catalog/popular`
Trending games by recent popularity (as of the engine clock).

| Query | Default | Meaning |
|---|---|---|
| `k` | 24 | 1-100 |
| `offset` | 0 | paging |
| `new_only` | false | only games marked new (no interactions before the cutoff) |

**How it works:** `Product.popular` walks `items.recent_pop_order`. Response: `{"items": [card, ...]}`.

### GET `/catalog/search`
Title search. Query: `q` (required), `k` (24, 1-100).

**How it works:** `Product.search`: tokens of `q` match title tokens exactly or as a prefix; more matching
tokens first, an exact substring match gets a bonus, ties broken by recent popularity.
Response: `{"query": q, "items": [card, ...]}`.

### GET `/catalog/similar/{item_id}`
Games closest to one game. Query: `k` (12, 1-50), `user_id` (optional: exclude this user's history).

**How it works:** `Product.similar`: cosine similarity of the two-tower item embeddings (the vectors FAISS
serves). Unknown item → `404`. Response: `{"item_id", "items": [card + "similarity", ...]}`.

### GET `/users/demo`
Real dataset players for "explore as a real player". Query: `k` (12, 1-50).

**How it works:** `Product.demo_users`: users with at least 8 games, most recent first (top 400), a fixed
shuffle (seed 7), so `k=9` is always a subset of `k=50` (the web app relies on that to validate a pick).
Response: `{"users": [{"user_id", "n_games", "recent": [3 titles]}]}`.

### GET `/users/{user_id}/history`
A user's most recent games, newest first. Query: `k` (20, 1-200).

**How it works:** `Product.history` reads the feature store, i.e. exactly the history the user tower sees.
Unknown user → empty list. Response: `{"user_id", "items": [card, ...]}`.

### GET `/health`
Liveness + what is being served. Used by the Docker healthcheck, the e2e global setup and the admin page.

Response fields: `status, category, store (redis|memory), users, items, t_ref, model_run, model_dir,
model_loaded_at, emb_version, index, index_knob, retrieval_variant, ranker_iterations, cold_user_fallback,
new_item_slots` (from `Recommender.info`) and `reloads, last_reload_error, watch_seconds, experiment,
experiment_error, experiment_warning` (from `ModelManager.info` / `ExperimentManager.info`).

### GET `/metrics`
Prometheus text format (scraped by `infra/monitoring/prometheus.yml`; multi-worker safe via
`PROMETHEUS_MULTIPROC_DIR`). Metrics: `recosphere_requests_total{source,arm,model_run}`,
`recosphere_request_seconds{arm}`, `recosphere_stage_seconds{stage}`, `recosphere_errors_total{kind}`,
`recosphere_top_score{model_run}`, `recosphere_new_item_share{arm}`, `recosphere_feedback_total{arm,action}`,
`recosphere_events_total{recorded}`, `recosphere_model_reloads_total{result}`,
`recosphere_model_info{model_run,emb_version}`. Alert rules: `infra/monitoring/alerts.yml`.

### POST `/admin/reload`
Reload now instead of waiting for the `MODEL_WATCH_SECONDS` poll. Query: `force` (false; reload even if
the pointer did not change or the target failed before).

**How it works:** `ModelManager.reload`: re-read `models/<cat>/production.json`, load the new model in the
background while the old one serves, smoke-test it, swap one reference. A failed load keeps the old model.
Then `ExperimentManager.refresh(force=True)` re-reads the experiment config.
Response: `{"model": {"status": "swapped|unchanged|skipped|failed|busy", ...}, "experiment": {...}}`.

### GET `/admin/summary`
Everything the app's admin page shows. Response: `{"health": <as /health>, "live": {"arms", "errors",
"reloads"} (from `metrics.snapshot`), "production_history": [last 15 promotions, newest first]}`.

### POST `/admin/rollback`
Point production at the previous model and hot-swap. **Changes `production.json` on disk.**

**How it works:** `pipeline/gate.py: rollback` promotes the newest history entry that is not the current
model (logged in `production_history.jsonl`), then `ModelManager.reload("admin rollback")`. Other workers
follow within `MODEL_WATCH_SECONDS`. Response: `{"rolled_back_to": run_id, "log": [lines], "swap": {...}}`.
No previous model → error (`RuntimeError: nothing to roll back to`).

## 2. Web app API (`web/app/api/*`, Next.js route handlers)

Called by the browser with the session cookie (`rs_session`, httpOnly JWT signed with `AUTH_SECRET`).
`web/middleware.ts` sends logged-out page requests to `/login` and answers logged-out `/api/*` calls (except
`/api/auth/*`) with `401 {"error": "unauthorized"}`. Accounts live in the app DB (`web/lib/db.ts`, SQLite now,
Postgres in P1).

| Method | Path | Purpose | Engine calls |
|---|---|---|---|
| POST | `/api/auth/signup` | create an account, sign in | – |
| POST | `/api/auth/login` | sign in | – |
| POST | `/api/auth/logout` | sign out | – |
| POST | `/api/auth/demo` | sign in as a real dataset player | `GET /users/demo?k=50` |
| POST | `/api/onboarding` | save the first picks (cold start) | `POST /event` per pick |
| POST | `/api/play` | start / end a play session | `POST /event`, `POST /feedback` (play) |
| POST | `/api/feedback` | click beacon from a recommendation tile | `POST /feedback` |
| GET | `/api/admin/summary` | admin dashboard data | `GET /admin/summary` |
| POST | `/api/admin/rollback` | admin "Roll back" button | `POST /admin/rollback` |

The pages themselves (home, game, search, library, onboarding) call the engine while rendering on the server:
`/recommend?meta=true`, `/explain`, `/users/{id}/history`, `/catalog/*`.

### POST `/api/auth/signup`
Body `{"email", "password", "name"?}`. Email must look valid, password 8+ characters. Creates an `app`
user (bcrypt hash, engine id `app:<id>`), sets the session cookie.
→ `{"ok": true, "next": "/onboarding"}`; `400` invalid input; `409` email exists.

### POST `/api/auth/login`
Body `{"email", "password"}`. → `{"ok": true, "next": "/" | "/onboarding"}` (onboarding until the first
picks are saved); `401 {"error": "Wrong email or password"}`.

### POST `/api/auth/logout`
Clears the cookie, returns `{"ok": true}`. The browser navigates to `/login` itself (a server redirect once
sent users to another origin and lost the cookie; see the comment in the route).

### POST `/api/auth/demo`
Body `{"user_id"}`. Accepted only if the engine offers that id in `/users/demo?k=50`; creates (once) a
`dataset` user mapped to it and signs in. → `{"ok": true, "next": "/"}`; `400 {"error": "unknown player"}`.

### POST `/api/onboarding`
Body `{"item_ids": [...]}`, at least 3 (max 20 used). Each pick → engine `POST /event`, so the first feed is
already personalized. Marks the user onboarded. → `{"ok": true}`; `400` fewer than 3; `401`.

### POST `/api/play`
Start: body `{"item_id", "rid"?}` → engine `POST /event` (history, feed changes at once) and, when the play
came from a recommendation (`rid` = its `request_id`), engine `POST /feedback` with `action: "play"`; stores
a play session. → `{"ok": true, "play_id", "first_time": bool}` (`first_time` = the engine's `recorded`).
End: body `{"end": true, "play_id", "seconds"}` (sent as a beacon on page hide) stores the session length.
`401` when logged out.

### POST `/api/feedback`
Body `{"request_id", "item_id", "action"?}` (`play`, anything else counts as `click`). Forwards to engine
`POST /feedback` with the session's engine user id; engine errors are swallowed (a lost click must not
break navigation). → `{"ok": true}`; `400` missing fields; `401`.

### GET `/api/admin/summary` · POST `/api/admin/rollback`
Admins only (email listed in `ADMIN_EMAILS`), else `403 {"error": "forbidden"}`. Forward to the engine's
`/admin/summary` and `/admin/rollback` with `X-Admin-Token: ENGINE_ADMIN_TOKEN` when set. Rollback errors
→ `500 {"error": ...}`.

## Change log
| Date | Change |
|---|---|
| 2026-11-15 | First version: 15 engine endpoints, 9 web API routes. |
