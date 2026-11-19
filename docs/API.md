# RecoSphere API reference

Every HTTP endpoint in the system: what it does, how to call it, and which code does the work.

**Keep this file in sync.** Any change that adds, changes or removes an endpoint updates this file in the
same commit. `tests/test_api_doc.py` fails CI when an engine route, a web `/api/*` route or an app_api
route is missing here (each section is checked against its own service).

```
browser ──► web app (Next.js, :3000)  /api/*  ── server-side fetch ──►  engine  (FastAPI, :8000)   ML serving
            pages render on the server             lib/engine.ts            Swagger: /docs, /redoc
                                                                    ──►  app_api (FastAPI, :8100)   user data on Postgres
                                                                         (P1, being built: docs/platform/P1_ROADMAP.md)
```

- **The browser never calls the engine.** It talks to the web app only. The web app calls the engine from
  the server (`web/lib/engine.ts`), so the engine needs no CORS and its admin token never reaches a browser.
- **Engine user ids:** app accounts are `app:<id>`; "real players" are dataset user ids (Amazon reviewer ids).
- **Engine clock:** frozen at the start of the test window (`t_ref` in `/health`, July 2021 on the real data).
  Events without `ts` are stamped with it. Real time comes in P4.
- **Live, interactive docs:** `http://127.0.0.1:8000/docs` (Swagger UI), `/redoc`, `/openapi.json`.
- **Item types** (`data/item_types.py`): every item is `game`, `accessory` (controllers, cases, consoles,
  figures ...) or `other` (gift cards, memberships, guides, non-gaming). Lists take an optional `item_type`
  filter applied AFTER ranking; without it the output is exactly the trained pipeline's. Types come from
  `data/gold/<cat>/item_types.parquet` (`python -m data.item_types --write`), else they are classified from the
  gold text at startup. Real catalog: 12,422 games, 13,045 accessories, 145 other.

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
| POST | `/users/{user_id}/history` | replay many interactions at once (after an engine restart) | `routers/events.py` |
| GET | `/users/{user_id}/fresh` | items new to the catalog, by the user's taste | `routers/catalog.py` |
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
| `meta` | false | return full game cards (as `/catalog/items`) instead of id/title/is_new/item_type |
| `item_type` | – | `game`, `accessory` or `other`: keep only that type (see *Item types* above); invalid value → `422` |
| `exclude` | – | comma-separated item ids to leave out (the user's "Not interested" games, at most 500, more → `422`); dropped after ranking, so the other items keep their order and scores |

**How it works** (`serve/recommender.py: Recommender.recommend`):
1. **A/B routing** (`serve/experiments.py: ExperimentManager.route`): salted sha256 of the user id picks the
   arm of the active experiment (`experiments/active.yaml`); each arm can be another model or policy.
2. **User state** from the feature store (`serve/feature_store.py`, Redis or in-memory): history, seen items.
3. **No history → `source: "fallback"`**: the popularity fallback list.
4. **Candidates** (`score_candidates`): user tower embeds the history (`retrieval/two_tower.py`) → FAISS top
   `n_tt` + recent-popularity top `n_pop`, seen items removed.
5. **Features + ranking**: `ranking/pipeline.py: assemble` builds the 22 features (the same code as training:
   parity), LightGBM LambdaRank scores them, ties broken by item id.
6. **Type filter + re-rank**: with `item_type`, other types are dropped from the ranked list (order kept);
   then `new_item_slots` makes sure up to `NEW_ITEM_SLOTS` (default 2) new items are in the top-k, taken from
   the top 100 of that list. If none of those 100 is new, fewer appear. A user without history gets the
   fallback list, filtered the same way.
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
    {"item_id": "B097PBVHYP", "title": "Xbox PC Wireless Controller – Space Jam: ...", "is_new": true, "item_type": "accessory", "score": 0.31845},
    {"item_id": "B0BPMJ8TRV", "title": "PowerA Spectra Infinity Enhanced Wired Controller ...", "is_new": true, "item_type": "accessory", "score": 0.17372}
  ],
  "timings_ms": {"store_get": 0.0, "user_tower": 0.0, "ann_candidates": 0.0, "features": 0.0, "ranker": 0.0, "rerank": 0.0, "total": 0.0},
  "request_id": "c58cf3a86cce4a22bad3688d10dd6c11",
  "experiment": "new-slots-2-vs-0",
  "arm": "control",
  "model_run": "20261111-171751"
}
```
(`timings_ms` values are placeholders here; they are real per request. Both items are controllers: this is
the unfiltered output for a real user, the reason the app always asks for `item_type=game`.) `source` is `personalized`,
`fallback` or `fallback_error`. `experiment` is `null` and `arm` is `control` when no experiment is active.

### GET `/explain/{user_id}`
Why games are in this user's feed. Used by the app's "Why this?" panel.

| Query | Default | Meaning |
|---|---|---|
| `items` | the user's served top 10 | comma-separated item ids to explain |
| `item_type` | – | the feed the user sees (`game` in the app): decides the served top 10 for `in_top10` and the default `items` |
| `exclude` | – | the user's "Not interested" games, as on `/recommend`: the served top 10 is computed without them |

**How it works** (`serve/product.py: Product.explain`), with the model and policy of the user's A/B arm:
re-scores the user's candidates and rebuilds the served top 10 exactly like `/recommend` (ranker order →
`item_type` filter → new-item slots); then for each item takes the LightGBM feature contributions
(SHAP, `booster.predict(pred_contrib=True)`), maps positive ones to plain-language reasons (`REASON` table;
user-only features are skipped because they do not separate items), keeps the top 3 groups, and finds the
closest game in the user's history by item-embedding cosine ("because you played").

Response: `{"user_id", "arm", "model_run", "source", "items": {<item_id>: {...}}}`. Per item:
`in_candidates`, `in_top10` (is it in the user's served top 10?), `rank` (ranker position among all
candidates), `n_candidates`, `score`, `sources` (`taste model` / `trending`), `because_you_played`,
`reasons: [{group, text, weight, feature, share, strength}]`. `weight` = raw SHAP contribution (admin only in
the app); `share` = weight / sum of the item's positive explainable contributions; `strength` = `strong`
(share ≥ 0.40), `medium` (≥ 0.20) or `weak` (thresholds are a product choice, `serve/product.py`). An item outside the candidate set gets
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
release `year`, `description` (max 600 chars), `is_new`, `item_type`, `plays_90d`, `trending_rank`.

Response: `{"items": [card, ...]}`. Card fields: `item_id, title, platforms, categories, year, description,
is_new, item_type, plays_90d, trending_rank`.

### GET `/catalog/popular`
Trending games by recent popularity (as of the engine clock).

| Query | Default | Meaning |
|---|---|---|
| `k` | 24 | 1-100 |
| `offset` | 0 | paging |
| `new_only` | false | only items marked new (no interactions before the cutoff) |
| `item_type` | – | `game`, `accessory` or `other`: keep only that type (see *Item types* above); invalid value → `422` |
| `exclude` | – | comma-separated item ids to leave out (the user's "Not interested" games, at most 500, more → `422`); dropped after ranking, so the other items keep their order and scores |

**How it works:** `Product.popular` walks `items.recent_pop_order`. Response: `{"items": [card, ...]}`.

### GET `/catalog/search`
Title search. Query: `q` (required), `k` (24, 1-100), `item_type` (optional filter, see *Item types*).

**How it works:** `Product.search`: tokens of `q` match title tokens exactly or as a prefix; more matching
tokens first, an exact substring match gets a bonus, ties broken by recent popularity.
Response: `{"query": q, "items": [card, ...]}`.

### GET `/catalog/similar/{item_id}`
Games closest to one game. Query: `k` (12, 1-50), `user_id` (optional: exclude this user's history),
`item_type` (optional filter, see *Item types*), `exclude` (optional: comma-separated item ids to leave out, the
user's "Not interested" games, at most 500).

**How it works:** `Product.similar`: cosine similarity of the two-tower item embeddings (the vectors FAISS
serves). Unknown item → `404`. Response: `{"item_id", "items": [card + "similarity", ...]}`.

### GET `/users/demo`
Real dataset players for "explore as a real player". Query: `k` (12, 1-50).

**How it works:** `Product.demo_users`: users with at least 8 games, most recent first (top 400), a fixed
shuffle (seed 7), so `k=9` is always a subset of `k=50` (the web app relies on that to validate a pick).
Response: `{"users": [{"user_id", "n_games", "recent": [3 titles]}]}`.

### GET `/users/{user_id}/history`
A user's most recent items, newest first. Query: `k` (20, 1-200), `item_type` (optional filter: e.g. the
last *game* played, skipping accessories bought since).

**How it works:** `Product.history` reads the feature store, i.e. exactly the history the user tower sees.
Unknown user → empty list. Response: `{"user_id", "items": [card, ...]}`.

### POST `/users/{user_id}/history`
Bulk `/event`: replays a list of interactions, **oldest first**, for one user. The web app calls it on login, on
demo login, and on the home page when an app user has no engine history, with the user's saved library from
Postgres (`web/lib/history.ts`): the engine's history is a cache that a restart without Redis loses, Postgres
is the source of truth (P1 phase F).

Body: `{"items": [{"item_id": str, "ts": int | null, "rating": float | null}, ...]}`, at most 500 items
(more → `422`). `ts` defaults to `t_ref` and `rating` to 5.0, as in `/event`.

**How it works:** `Recommender.add_events` calls `add_event` for each item in order, so dedup, `hist_len`
trimming and counts are exactly those of the same `/event` calls (unit test `test_replay_equals_one_by_one_events`).
Replaying twice records nothing new. Unknown items are skipped, not an error (the catalog may have changed).
Counted in `recosphere_events_total{recorded}` and `recosphere_errors_total{unknown_item}`.

Response: `{"user_id", "recorded": int, "already_seen": int, "unknown": [item_id, ...]}`.

### GET `/users/{user_id}/fresh`
"Fresh on RecoSphere": items new to the catalog (`is_new`: no interaction before the cutoff; this is NOT a
release date, a 2016 game can be new here), without the ones the user has already played.
Query: `k` (20, 1-100), `item_type` (optional filter; the app sends `game`), `exclude` (optional: comma-separated
item ids to leave out, the user's "Not interested" games, at most 500).

**How it works** (`Product.fresh`): with history, sorted by taste = user-tower vector · item vector (the
model's `tt_score`, ties by item id). The two-tower model is only run, not changed, and the ranker is not used:
its strongest features are popularity windows, all 0 for new items. Without history: newest first by
first-available date (unknown dates last), because popularity cannot order items nobody has played.
Real catalog: 343 new games, 2,406 new accessories, 11 new other.
Response: `{"user_id", "ranked_by": "taste" | "newest", "items": [card (+ "score" when by taste), ...]}`.

### GET `/health`
Liveness + what is being served. Used by the Docker healthcheck, the e2e global setup and the admin page.

Response fields: `status, category, store (redis|memory), users, items, t_ref, model_run, model_dir,
model_loaded_at, emb_version, index, index_knob, retrieval_variant, ranker_iterations, cold_user_fallback,
new_item_slots, item_types` (counts per type), `item_types_source` (from `Recommender.info`) and `reloads, last_reload_error, watch_seconds, model_source
(local|s3), production_version, experiment, experiment_error, experiment_warning` (from `ModelManager.info` /
`ExperimentManager.info`). With `MODEL_REGISTRY=s3` (P2 phase C) `model_dir` is the verified copy in
`MODEL_CACHE_DIR`, `production_version` the S3 pointer's version, and `last_reload_error` names a refused run
(unfinished upload, a file whose sha256 differs, a manifest that is not the promoted one) while the previous model
keeps serving.

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
Point production at the previous model and hot-swap. **Changes the production pointer**: `production.json` on
disk, or with `MODEL_REGISTRY=s3` the S3 pointer this engine follows (same `MODEL_NAMESPACE`; compare-and-swap).

**How it works:** `pipeline/gate.py: rollback` promotes the newest history entry that is not the current
model (local `production_history.jsonl` or S3 `history/`), then `ModelManager.reload("admin rollback")`. Other workers
follow within `MODEL_WATCH_SECONDS`. Response: `{"rolled_back_to": run_id, "log": [lines], "swap": {...}}`.
No previous model → error (`RuntimeError: nothing to roll back to`).

## 2. Web app API (`web/app/api/*`, Next.js route handlers)

Called by the browser with the session cookie (`rs_session`, httpOnly JWT signed with `AUTH_SECRET`, carrying
`sid` = the app_api login session id). `web/middleware.ts` (edge, no database) only checks the signature: it sends
logged-out page requests to `/login` and answers logged-out `/api/*` calls (except `/api/auth/*`) with
`401 {"error": "unauthorized"}`. Whether the session is still alive is checked on the server through app_api
`GET /auth/me` (once per request, `web/lib/auth.ts`). User data lives in Postgres behind app_api (section 3,
client `web/lib/api.ts`); cookies from before P1 (`uid` claim) are treated as logged out.

| Method | Path | Purpose | Calls |
|---|---|---|---|
| POST | `/api/auth/signup` | create an account, sign in | app_api `POST /auth/signup` |
| POST | `/api/auth/login` | sign in | app_api `POST /auth/login`, `GET /library`, engine `POST /users/{id}/history` |
| POST | `/api/auth/logout` | sign out (revokes the session) | app_api `POST /auth/logout` |
| POST | `/api/auth/demo` | sign in as a real dataset player | engine `GET /users/demo?k=50`, app_api `POST /auth/demo`, `GET /library`, engine `POST /users/{id}/history` |
| POST | `/api/onboarding` | save the first picks (cold start) | app_api `POST /library/picks`, engine `POST /event` per pick |
| POST | `/api/play` | start / end a play session | app_api `POST /plays/start` / `end`, engine `POST /event`, `POST /feedback` (play) |
| POST | `/api/feedback` | click beacon from a recommendation tile | `POST /feedback` |
| GET | `/api/admin/summary` | admin dashboard data | `GET /admin/summary` |
| GET | `/api/admin/explain` | admin "Why-this inspector": raw explanation | `GET /explain/{user_id}?items=&item_type=game` |
| POST | `/api/admin/rollback` | admin "Roll back" button | engine `GET /health`, `POST /admin/rollback`, app_api `POST /admin/audit` |
| GET | `/api/admin/audit` | admin dashboard "Audit log" table | app_api `GET /admin/audit?k=20` |
| POST | `/api/not-interested` | hide a game from every recommendation row | app_api `PUT /feedback/{item_id}` |
| DELETE | `/api/not-interested` | undo "Not interested" | app_api `DELETE /feedback/{item_id}` |

The pages themselves (home, game, search, library, onboarding) call the engine while rendering on the server:
`/recommend?meta=true`, `/explain`, `/users/{id}/history`, `/users/{id}/fresh`, `/catalog/*`; the library page
reads app_api `GET /library` + `/library/stats` (cards from engine `/catalog/items`). When an app user has no engine history, the
home page replays their library (app_api `GET /library` → engine `POST /users/{id}/history`) and loads again.
A failed replay never fails a login or a page (`replayLibrarySafe`).

### POST `/api/auth/signup`
Body `{"email", "password", "name"?}`. Email must look valid, password 8+ characters (app_api also enforces
at most 72 bytes). app_api creates the user (engine id `app:<uuid>`) and a login session; the cookie gets its id.
→ `{"ok": true, "next": "/onboarding"}`; `400` invalid input; `409` email exists.

### POST `/api/auth/login`
Body `{"email", "password"}`. → `{"ok": true, "next": "/" | "/onboarding"}` (onboarding until the first
picks are saved); `401 {"error": "Wrong email or password"}`.

### POST `/api/auth/logout`
Revokes the login session in app_api (a copy of the old cookie no longer works; e2e checks this), clears the
cookie, returns `{"ok": true}`. The browser navigates to `/login` itself (a server redirect once sent users to
another origin and lost the cookie; see the comment in the route).

### POST `/api/auth/demo`
Body `{"user_id"}`. Accepted only if the engine offers that id in `/users/demo?k=50`; app_api creates (once) a
demo user mapped to it and a login session. → `{"ok": true, "next": "/"}`; `400 {"error": "unknown player"}`.

### POST `/api/onboarding`
Body `{"item_ids": [...]}`, at least 3 (max 20 used). Saved through app_api (`library`, source `onboarding`,
the user onboarded; `400` if one is not an active game), then each pick → engine `POST /event`, so the first feed
is already personalized. → `{"ok": true}`; `400` fewer than 3 or not games; `401`.

### POST `/api/play`
Start: body `{"item_id", "client_event_id", "rid"?, "surface"?}`. `client_event_id` is a uuid made by the browser
(`components/Player.tsx`); a repeated start stores one play. app_api stores the play + library row (one statement),
then engine `POST /event` (history, the feed changes at once) and, when the play came from a recommendation
(`rid` = its `request_id`), engine `POST /feedback` with `action: "play"`.
→ `{"ok": true, "play_id", "first_time": bool}` (`first_time` from Postgres: the game was not in the library).
End: body `{"end": true, "client_event_id", "seconds"}` (sent as a beacon on page hide) stores the length.
`400` without a uuid `client_event_id` or for a non-game, `404` unknown item, `401` when logged out.

### POST `/api/feedback`
Body `{"request_id", "item_id", "action"?}` (`play`, anything else counts as `click`). Forwards to engine
`POST /feedback` with the session's engine user id; engine errors are swallowed (a lost click must not
break navigation). → `{"ok": true}`; `400` missing fields; `401`.

### GET `/api/admin/explain`
Admins only (`403` otherwise). Query `user_id` (engine user id, e.g. `app:3` or a dataset id) and `item_id`,
both required (`400`). Returns the engine's `/explain` for that one item with the `game` feed as context,
including the raw SHAP `weight`, `rank`, `n_candidates` and `score` that the user-facing "Why this?" panel
hides (users only see strength bars). Engine errors → `502 {"error": ...}`.

### GET `/api/admin/summary` · POST `/api/admin/rollback`
Admins only (`users.role = 'admin'` in Postgres), else `403 {"error": "forbidden"}`. Forward to the engine's
`/admin/summary` and `/admin/rollback` with `X-Admin-Token: ENGINE_ADMIN_TOKEN` when set. Rollback errors
→ `500 {"error": ..., "audited"}`. Every rollback attempt, failed ones too, writes one app_api audit row
(`action: "model.rollback"`, `target` = new run, `details: {from_run, to_run, ok, error?}`); the response carries
`audited: bool` (a failed audit write does not undo a done rollback; the dashboard says the write failed).

### GET `/api/admin/audit`
Admins only, else `403`. The last 20 admin actions from app_api `GET /admin/audit` (newest first):
`{"items": [{"id", "actor_id", "action", "target", "details", "created_at"}, ...]}`. Shown as the
"Audit log" table on `/admin`.

### POST `/api/not-interested` · DELETE `/api/not-interested`
Body `{"item_id"}`. POST saves "Not interested" for the signed-in user (app_api `PUT /feedback/{item_id}` with
`kind: "not_interested"`); DELETE undoes it (Undo on the game page). → `{"ok": true, "hidden": bool}`;
`400` without `item_id` or for an unknown game; `401` when logged out. The pages read the list once per request
(`web/lib/hidden.ts`, app_api `GET /feedback/hidden`; one extra round trip) and send it as `exclude` to the
engine's `/recommend`, `/explain`, `/catalog/popular`, `/catalog/similar` and `/users/{id}/fresh`: the game
leaves every row (hero, top picks, because you played, fresh, trending, more like this). Search still finds it
(the user asked for it). If app_api fails, the rows are shown unfiltered rather than not at all.

## 3. App API (`app_api/`, FastAPI + SQLAlchemy)

User-owned data on Postgres (Neon): accounts, sessions, library, plays, feedback, admin audit. Classic ORM
layers: `routers/` → `services/` → `models/` (SQLAlchemy) → `db/` (engine, pool); `schemas/` = Pydantic.
Tables: `DB_SCHEMA.md`. Plan: `docs/platform/P1_ROADMAP.md`. Run: `make app-api` (port 8100), Swagger on `/docs`.

**Who may call it.** Only the web app, from its server. Every route except `/health` needs the header
`X-Service-Token: <SERVICE_TOKEN>` (constant-time check; with no token configured every protected route answers
`401`: fail closed). **Which user:** routes marked *session* also need `X-Session-Id: <session uuid>`, the
`session_id` returned by signup/login/demo that the web app keeps in its signed cookie (JWT claim `sid`).
Missing, unknown, revoked or expired session → `401`. Admin routes need `users.role = 'admin'` (`403` otherwise).

**Errors.** Validation: `422` (FastAPI format). Service errors: `{"detail": "...", "code": "..."}` with
`404 not_found`, `409 conflict`, `422 invalid`, `401 unauthorized`.

**Sessions per route** (latency): reads and single-statement writes run on an autocommit session (one round trip
each); multi-write routes run in one transaction. Measurements: `docs/platform/P1_ROADMAP.md`.

| Method | Path | Purpose | Router |
|---|---|---|---|
| GET | `/health` | database reachable? migration revision | `routers/health.py` |
| POST | `/auth/signup` | create an account + a login session | `routers/auth.py` |
| POST | `/auth/login` | email + password → a login session | `routers/auth.py` |
| POST | `/auth/demo` | sign in as a real dataset player | `routers/auth.py` |
| POST | `/auth/logout` | revoke the login session | `routers/auth.py` |
| GET | `/auth/me` | the session's user (every page) | `routers/auth.py` |
| POST | `/library/picks` | onboarding picks → library | `routers/library.py` |
| GET | `/library` | the user's recent games | `routers/library.py` |
| GET | `/library/stats` | games, plays, seconds | `routers/library.py` |
| POST | `/plays/start` | start a stream (idempotent) | `routers/plays.py` |
| POST | `/plays/end` | end a stream (idempotent) | `routers/plays.py` |
| PUT | `/feedback/{item_id}` | like / dislike / not_interested | `routers/feedback.py` |
| DELETE | `/feedback/{item_id}` | remove the feedback | `routers/feedback.py` |
| GET | `/feedback/hidden` | item ids every row must hide | `routers/feedback.py` |
| POST | `/admin/audit` | record an admin action | `routers/admin.py` |
| GET | `/admin/audit` | recent admin actions | `routers/admin.py` |

### GET `/health`
Runs `SELECT 1` on a pooled connection and reads `alembic_version`.
Response: `{"status": "ok", "database": "ok", "schema_revision": "0001_baseline", "db_roundtrip_ms": <float>}`
(`schema_revision` is `null` on a database that was never migrated or stamped). Database unreachable
(Neon suspended for too long, wrong credentials): `503 {"status": "error", "database": "<driver error>"}`.
`db_roundtrip_ms` = the first statement of a request: `BEGIN` + `SELECT 1`, two round trips. Measured
2026-11-17 from the laptop to Neon us-east-1 with a warm pool: 529-542 ms (one round trip is about 266 ms).

### POST `/auth/signup`
Body `{"email", "password", "display_name"?}`: email pattern, password 8-72 **bytes** (bcrypt >= 5 refuses
longer), `display_name` 1-40 chars (default: the part of the email before `@`). Optional header
`X-Client-User-Agent` (the browser's, stored on the session). Creates a `registered` user (bcrypt hash) and a
30-day login session in one transaction. → `201 SessionOut`: `{"session_id", "expires_at", "needs_onboarding",
"user": {"id", "kind", "email", "display_name", "role", "dataset_user_id", "onboarded_at", "created_at"}}`.
Same email in any case → `409 conflict`. (`auth_service.signup`)

### POST `/auth/login`
Body `{"email", "password"}`. → `200 SessionOut` (a new session; `last_login_at` updated). Wrong email or
password → `401 unauthorized` (an unknown email still runs a bcrypt check: no timing hint). (`auth_service.login`)

### POST `/auth/demo`
Body `{"dataset_user_id"}`. Creates the demo player once (`kind = demo`, already onboarded) and a session.
The CALLER must check that the engine knows this id (the web app checks it against the engine's `/users/demo`).
→ `200 SessionOut`. (`auth_service.demo_login`)

### POST `/auth/logout`
Session id header (`X-Session-Id`). Revokes that session: from now on it is `401` everywhere, which a stateless
JWT alone could not do. → `{"revoked": true}`; already revoked / unknown → `{"revoked": false}`.

### GET `/auth/me`
*Session.* → `200 SessionOut` of the active session. The hot path (every page): one autocommit query
(session + user join).

### POST `/library/picks`
*Session.* Body `{"item_ids": [...]}`: 3-20 unique ids, each an active `game`/`dlc` in `games`, else
`422 invalid` and nothing is written. Adds them to `library` (`source = onboarding`) and sets `onboarded_at`, in
one transaction. → `{"added": [ids new to the library]}`. (`library_service.add_picks`)

### GET `/library`
*Session.* Query `k` (20, 1-200), `games_only` (true: hide accessories bought on Amazon). Most recent first
(last play, else when added) → `[{"item_id", "title", "source", "added_at", "last_played_at", "play_count",
"seconds_played"}]`. This is the history the app sends to the engine.

### GET `/library/stats`
*Session.* → `{"games", "plays", "seconds_played"}`.

### POST `/plays/start`
*Session.* Body `{"item_id", "client_event_id", "request_id"?, "surface"?}`. `client_event_id` (uuid) is made
by the browser: a retried request returns the original answer and writes nothing. `request_id` (the engine
list it came from) and `surface` (e.g. `home:top_picks`) attribute the play. One statement (CTEs): the play row,
the library upsert and `first_time`. → `{"play_id", "item_id", "started_at", "ended_at", "seconds_played",
"first_time"}`. Unknown item → `404`; accessory / inactive → `422 invalid`; someone else's `client_event_id` →
`409`. (`play_service.start`)

### POST `/plays/end`
*Session.* Body `{"client_event_id", "seconds"}` (0-86,400). Sets `ended_at` + `seconds_played` once and adds the
seconds to the library row; a repeated end changes nothing. → the same shape as start. Unknown → `404`.

### PUT `/feedback/{item_id}`
*Session.* Body `{"kind"}`: `like` | `dislike` | `not_interested`. One current value per (user, game).
→ `{"item_id", "kind", "created_at"}`. Unknown item → `404`.

### DELETE `/feedback/{item_id}`
*Session.* → `{"cleared": bool}`.

### GET `/feedback/hidden`
*Session.* → `{"item_ids": [...]}`: games with `dislike` or `not_interested`, which every row must drop.

### POST `/admin/audit`
*Session, admin.* Body `{"action", "target"?, "details"?}`; `action` looks like `model.rollback` (area.verb).
→ `201 {"id", "actor_id", "action", "target", "details", "created_at"}`.

### GET `/admin/audit`
*Session, admin.* Query `k` (50, 1-500). Newest first.

The first admin is created outside HTTP: `ADMIN_PASSWORD=... python -m app_api.cli create-admin --email <email>`
(creates it, or promotes an existing account).

## Change log
| Date | Change |
|---|---|
| 2026-11-15 | First version: 15 engine endpoints, 9 web API routes. |
| 2026-11-17 | P1 phase A: new service app_api (section 3) with `GET /health`. |
| 2026-11-17 | P1 phase D: app_api auth, library, plays, feedback and admin audit routes; service token + session id headers. |
| 2026-11-19 | P2 phase C: `/health` gets `model_source` + `production_version`; with `MODEL_REGISTRY=s3` the engine serves the S3 pointer's run (fetched + verified) and `/admin/rollback` moves the S3 pointer. |
| 2026-11-18 | P1 phase G: engine `exclude` on `/recommend`, `/explain`, `/catalog/popular`, `/catalog/similar`, `/users/{id}/fresh`; web `POST`/`DELETE /api/not-interested`, `GET /api/admin/audit`; rollback is audited. |
| 2026-11-17 | P1 phase F: new engine `POST /users/{user_id}/history` (bulk replay); web login, demo login and home replay the library into the engine. |
| 2026-11-17 | P1 phase E: the web routes use app_api (Postgres) instead of SQLite; cookie carries `sid`; `/api/play` needs `client_event_id`. |
| 2026-11-15 | P0.5 item 3: new `GET /users/{user_id}/fresh` ("Fresh on RecoSphere" row). |
| 2026-11-15 | P0.5 item 2: `/explain` gets `item_type`, `in_top10`, reason `share` + `strength`; web `/api/admin/explain`. |
| 2026-11-15 | P0.5 item 1: `item_type` filter on `/recommend`, `/catalog/popular`, `/catalog/search`, `/catalog/similar`, `/users/{id}/history`; `item_type` in cards and `/health`. |
