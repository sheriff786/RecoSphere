# P1 roadmap: app data on Postgres (Neon) through a Python ORM

Status: decisions approved 2026-11-17. **Phases A-E done**: the web app's accounts, sessions, onboarding, plays
and library run on app_api + Neon (SQLite code removed); F-H open.

Phase E (2026-11-17): `web/lib/api.ts` (app_api client), cookie JWT carries `sid`, `/auth/me` once per request;
signup/login/logout/demo/onboarding/play/library through app_api; `web/lib/db.ts` deleted. e2e runs every time on
a THROWAWAY database (`scripts/e2e_db.py`: CI's Postgres service, or Neon's server from the laptop; dropped at the
end). Verified: synthetic e2e 6/6 (incl. "a revoked session's old cookie no longer logs in" and the library from
Postgres); real stack (real model, real Neon) 4/4 non-admin tests, the test users deleted afterwards. Compose:
new `app-api` service + `Dockerfile.app_api` (config validated; the images were NOT built: Docker is not running
on the laptop).

**Re-cut on 2026-11-17 (approved):** D and E as first planned were coupled: a user created in Neon cannot onboard
or play while those still live in SQLite (integer user ids, foreign keys). So **D = all app_api HTTP routes** (auth,
library, plays, feedback, admin audit; the web app still on SQLite, nothing breaks) and **E = the web app switches
to app_api in one go** (login, onboarding, plays). The service token (X-Service-Token) and the CLI-created first
admin were approved with it.

Measured after D (2026-11-17, laptop -> Neon us-east-1, real HTTP through uvicorn, keep-alive client; one
throwaway user, deleted afterwards): `GET /auth/me` 292 ms median of 7 (1 round trip), `GET /library` 603 ms
(2), `POST /plays/start` 605 ms (2: session + the single statement), `POST /auth/login` 2,469 ms median of 3
(~6 round trips + bcrypt: too slow, to be folded into fewer statements later), single samples: signup 1,874 ms,
`/library/picks` 3,679 ms (opened a new connection too), `/plays/end` 1,790 ms (sent as a beacon on page hide). The schema itself was applied on Neon in commit `ca015cd`.

Legend: 🆕 new · ✏️ modified · 🗑️ removed · ✅ exists, unchanged

## Goal

User-owned data (accounts, sessions, library, plays, feedback, admin audit) moves from the web app's SQLite
file to the 7 tables on Neon, behind a **Python service built the classic ORM way**:

| Layer | Folder | Job |
|---|---|---|
| DB connection | `app_api/db/` | SQLAlchemy engine + connection pool, one session per request (`get_db`) |
| Model classes | `app_api/models/` | one SQLAlchemy class per table; relationships as attributes (`user.library`) |
| Schemas | `app_api/schemas/` | Pydantic models: what a request may contain and what a response returns |
| Services | `app_api/services/` | business rules and transactions; the only layer that uses models |
| Routers | `app_api/routers/` | HTTP endpoints: parse a schema, call a service, return a schema |

Also fixed on the way: today the engine forgets every app user's plays when it restarts. After P1 the
`library` table is the source of truth and is replayed into the engine.

## Architecture

```
browser ──► web (Next.js) ──server-side──► app_api (FastAPI + SQLAlchemy) ──► Neon Postgres (-pooler host)
                 │                               │
                 │                               └──► engine: POST /users/{id}/history (replay library)
                 └──server-side──► engine (FastAPI, ML): /recommend, /explain, /catalog/* (unchanged)
```

- **`app_api` is a separate, light service** (FastAPI + SQLAlchemy + Pydantic, no torch/FAISS/LightGBM).
  The engine stays ML serving only and never connects to Postgres.
- **The browser still never calls a backend directly**: the web app calls both services from the server, like
  it calls the engine today. Neither service needs CORS.
- **Login**: `app_api` checks the password and creates the `sessions` row; the web app keeps the signed
  cookie (JWT with `sid`) and asks `app_api` who the session belongs to.

## Files and folders

### The ORM service (all new)

```
app_api/                                   🆕  account & library API (port 8100)
├── __init__.py
├── main.py                                create the FastAPI app, include routers, lifespan (pool open/close)
├── config.py                              settings from env (pydantic-settings): DATABASE_URL, DATABASE_URL_DIRECT,
│                                          AUTH_SECRET, ENGINE_URL, SERVICE_TOKEN
├── security.py                            bcrypt hashing, JWT (sid) sign/verify, service-token check
├── deps.py                                FastAPI dependencies: get_db, current_user, require_admin
├── db/                                    "python db connect"
│   ├── __init__.py
│   ├── base.py                            class Base(DeclarativeBase) + naming convention for constraints
│   └── session.py                         create_engine(postgresql+psycopg://..., pool_size, pool_pre_ping,
│                                          prepare_threshold=None for the PgBouncer pooler), SessionLocal, get_db()
├── models/                                "model classes": one SQLAlchemy class per table
│   ├── __init__.py                        imports every model (so Alembic sees all tables)
│   ├── enums.py                           UserKind, UserRole, ItemType, LibrarySource, FeedbackKind (Python Enums
│   │                                      mapped to the existing Postgres enum types)
│   ├── user.py                            class User        -> users          (sessions, library, plays, feedback)
│   ├── session.py                         class Session     -> sessions       (user)
│   ├── game.py                            class Game        -> games
│   ├── library.py                         class LibraryItem -> library        (user, game)
│   ├── play.py                            class PlaySession -> play_sessions  (user, game)
│   ├── feedback.py                        class ItemFeedback-> item_feedback  (user, game)
│   └── audit.py                           class AdminAuditLog -> admin_audit_log (actor)
├── schemas/                               "schema": Pydantic request/response models
│   ├── __init__.py
│   ├── auth.py                            SignupIn, LoginIn, DemoLoginIn, SessionOut
│   ├── user.py                            UserOut, MeOut
│   ├── game.py                            GameOut
│   ├── library.py                         PicksIn, LibraryItemOut, LibraryStatsOut
│   ├── play.py                            PlayStartIn (client_event_id, request_id, surface), PlayEndIn, PlayOut
│   ├── feedback.py                        FeedbackIn, FeedbackOut
│   └── audit.py                           AuditIn, AuditOut
├── services/                              "services": business rules + transactions
│   ├── __init__.py
│   ├── auth_service.py                    signup, login, demo login, logout (revoke), session -> user
│   ├── library_service.py                 onboarding picks, recent history, stats
│   ├── play_service.py                    start = play_sessions + library upsert in ONE transaction;
│   │                                      same client_event_id again = no-op; end = seconds + ended_at
│   ├── feedback_service.py                set like / dislike / not_interested; hidden item ids
│   ├── catalog_service.py                 upsert games from gold (used by scripts/load_catalog.py)
│   ├── audit_service.py                   record admin actions, list recent
│   └── engine_sync.py                     replay a user's library into the engine when it has none
└── routers/                               HTTP endpoints (thin)
    ├── __init__.py
    ├── auth.py                            POST /auth/signup, /auth/login, /auth/demo, /auth/logout, GET /auth/me
    ├── library.py                         POST /library/picks, GET /library, GET /library/stats
    ├── plays.py                           POST /plays/start, POST /plays/end
    ├── feedback.py                        PUT /feedback/{item_id}, GET /feedback/hidden
    ├── admin.py                           POST /admin/audit, GET /admin/audit
    └── health.py                          GET /health (DB reachable, migration revision)
```

### Migrations (Alembic, decision 2)

```
alembic.ini                                🆕  points at app_api/migrations
app_api/migrations/                        🆕
├── env.py                                 target_metadata = Base.metadata; URL = DATABASE_URL_DIRECT
├── script.py.mako
└── versions/
    └── 0001_baseline.py                   runs web/db/migrations/001_init.sql as is (already on Neon:
                                           `alembic stamp 0001` there, `alembic upgrade head` elsewhere)
```

From then on a schema change = edit the model → `alembic revision --autogenerate` → review → `alembic upgrade`.

### Tests

```
tests/app_api/                             🆕  pytest against a real Postgres (CI service container; never Neon)
├── conftest.py                            fresh schema per test session, one transaction per test (rolled back)
├── test_models.py                         models match the database: Alembic autogenerate finds NO diff
├── test_auth_service.py                   signup, duplicate email (case), wrong password, logout revokes
├── test_play_service.py                   one transaction, idempotent client_event_id, negative seconds refused
├── test_library_service.py                picks, most recent first, stats
├── test_feedback_service.py               one current value, hidden ids
├── test_routers.py                        HTTP level with TestClient: status codes, schema validation (422)
└── test_engine_sync.py                    replay sends the library once, not when the engine already has it
```

### Existing files that change

```
requirements-app.txt                       🆕  fastapi, uvicorn, sqlalchemy>=2, psycopg[binary]>=3, alembic,
                                               pydantic-settings, bcrypt, pyjwt (no ML packages)
requirements-ci.txt                        ✏️  + -r requirements-app.txt
Dockerfile.app_api                         🆕  python:3.11-slim + requirements-app.txt + app_api/ (small image)
infra/compose/docker-compose.yml           ✏️  new core service app-api (8100); web gets APP_API_URL;
                                               web-data (SQLite) volume removed
Makefile                                   ✏️  app-api (run locally), db-upgrade, db-revision, load-catalog
.github/workflows/ci.yml                   ✏️  db job: alembic upgrade + tests/app_api; e2e job: Postgres service
scripts/load_catalog.py                    🆕  gold + item_types + title_family -> games via catalog_service
data/title_family.py                       🆕  normalized title family (also P0.5 item 4: edition dedup)
serve/routers/events.py                    ✏️  POST /users/{user_id}/history: replay many events at once
serve/recommender.py, serve/feature_store.py ✏️ bulk add with the same dedup rule as /event
docs/API.md                                ✏️  new section "3. App API (app_api)" + the engine's replay endpoint
                                               (tests/test_api_doc.py extended to app_api routes)
DB_SCHEMA.md                               ✏️  Alembic + models; "app still on SQLite" removed
web/lib/api.ts                             🆕  typed client for app_api (like web/lib/engine.ts), server-side only
web/lib/auth.ts, web/lib/session.ts        ✏️  cookie carries sid; currentUser() = GET /auth/me on app_api
web/app/api/*/route.ts                     ✏️  signup, login, logout, demo, onboarding, play, feedback, admin ->
                                               call app_api instead of web/lib/db.ts
web/app/page.tsx, web/app/library/page.tsx ✏️  history and library from app_api
web/components/Player.tsx                  ✏️  sends a client_event_id with every start and end
web/components/GameCard.tsx                ✏️  "Not interested" action
web/components/AdminDashboard.tsx          ✏️  audit log table
web/lib/db.ts, better-sqlite3              🗑️  SQLite gone
web/db/migrate.mjs, web/db/check_schema.mjs 🗑️ replaced by Alembic + tests/app_api (001_init.sql stays as the
                                               file the baseline revision runs)
web/Dockerfile, web/next.config.mjs        ✏️  no native SQLite module
web/e2e/app.spec.ts                        ✏️  new cases per phase
```

## Phases

One phase = one commit, CI green, and a check with the owner before the next one.

| # | Phase | Builds | Done when |
|---|---|---|---|
| **A** | Skeleton + DB connect + models | `app_api/` main, config, `db/`, `models/`, Alembic baseline; stamp Neon at `0001`; `/health` | `test_models.py`: Alembic autogenerate against Neon's schema finds **no** difference; `/health` OK on Neon |
| **B** | Schemas + services | `schemas/`, `services/` (auth, library, play, feedback, audit), unit tests | `tests/app_api` green in CI; play transaction + idempotency proven |
| **C** | Catalog | `data/title_family.py`, `catalog_service`, `scripts/load_catalog.py` (3 → 5 type mapping: game, accessory, other; console/dlc unused until the classifier splits them) | Neon `games` = 25,612 rows; type counts = the classifier's (12,422 / 13,045 / 145 on 2026-11-15) |
| **D** | Routers + auth in the app | `routers/`, `security.py`, `web/lib/api.ts`; web signup/login/logout/demo through app_api; JWT `sid` | e2e: signup → login → logout → the old cookie is rejected (a stateless JWT could not do this) |
| **E** | Library + plays | onboarding and play through app_api; Player sends `client_event_id`; library page from Postgres | e2e green; a retried play start creates one row; `request_id` + `surface` stored |
| **F** | Engine sync | engine `POST /users/{id}/history`; `engine_sync` replays the library on login | e2e (synthetic): play, restart the engine, log in → feed still personalized |
| **G** | Feedback + audit | "Not interested" → `item_feedback` (hidden in rows); rollback → `admin_audit_log` | e2e: the game disappears; the rollback leaves an audit row |
| **H** | Cutover | remove SQLite + the Node migration scripts; compose `app-api` service; CI e2e on Postgres; docs | every CI job green; no SQLite left |

## Rules for every phase

- **Layers only call downward**: routers → services → models → db. Routers never touch models; services
  never build HTTP responses; models hold no business rules.
- **A transaction belongs to a service method**, not a router (play = `play_sessions` + `library`).
- **Schema changes go through models + Alembic**, never by hand on Neon. `test_models.py` fails when the
  models and the database disagree.
- **Two URLs**: the app uses Neon's `-pooler` host (`DATABASE_URL`), Alembic the direct host
  (`DATABASE_URL_DIRECT`). Both only in git-ignored env files; CI uses its own Postgres and never sees Neon.
- **PgBouncer**: psycopg's automatic prepared statements are switched off (`prepare_threshold=None`), else
  queries through the pooler fail.
- **Latency budget**: Neon us-east-1 costs about 266 ms per round trip from the laptop (measured 2026-11-15
  and 2026-11-17). At most 2 sequential round trips per request (eager-load relationships with
  `selectinload`/joins, no N+1 queries); every phase measures request time before and after.
  Measured in phase A: `pool_pre_ping` added a round trip to every request (first statement 813 ms with it,
  532 ms without), so it is off and `pool_recycle=240` covers Neon's 5-minute suspend instead. A transaction's
  `BEGIN` is the second round trip of every request.
  Decided in phase B (measured 2026-11-17, laptop -> Neon, warm pool, median of 7, nothing written):
  - **Reads use an autocommit session** (`db/session.py: get_autocommit_db`): session -> user 263-278 ms
    instead of 785-828 ms in a normal transaction (BEGIN + SELECT + ROLLBACK).
  - **Play start is ONE statement** (`play_service.start_statement`: data-modifying CTEs insert the play,
    upsert the library and compute `first_time`). One statement is atomic by itself, so it may run on the
    autocommit session: 1 round trip (estimate; measured inside savepoints: 1,573 ms with the first
    6-statement version, 833 ms now). `tests/app_api/test_play_service.py` pins it to exactly one statement
    and proves atomicity in the database (a failing library trigger leaves no play row).
  - Login/signup stay multi-statement transactions (rare, and bcrypt itself costs about 0.25 s).
- **Every endpoint** is documented in `docs/API.md` in the same commit (the doc test will cover app_api).

## Decisions

| # | Question | Proposed | Why |
|---|---|---|---|
| 1 | Where the ORM code runs | **separate service `app_api/`** | the engine is ML serving (heavy models, many workers, hot swap); account data has a different load, scaling and release cycle |
| 2 | Migrations | **Alembic, with `001_init.sql` as the baseline revision** | the classic SQLAlchemy way: models are the source of truth from now on; Neon is stamped, not rebuilt |
| 3 | Driver | **SQLAlchemy 2.0, sync sessions, psycopg 3** | simple, well supported; FastAPI runs sync endpoints in a thread pool; async (asyncpg) needs extra PgBouncer workarounds |
| 4 | Existing SQLite accounts (local test users) | **drop, do not migrate** | test data only; ids change from integers to uuids |
| 5 | Order | **A → H**; P0.5 item 4 inside C; P0.5 items 5-6 after P1 | C needs `title_family`, which is item 4's normalization |

## Risks

| Risk | Mitigation |
|---|---|
| ~264 ms per query to us-east-1 | round-trip budget, eager loading; consider a Neon project in Singapore (not measured yet) |
| A third service to run | it is small (no ML); `make up` starts it; compose healthcheck |
| Free tier suspends when idle: slow first query | `pool_pre_ping`; `/health` and e2e global setup tolerate a slow first query |
| Models drifting from the database | `test_models.py` (Alembic autogenerate = no diff) in CI |
| App and engine disagree on a user's history | phase F replays `library`; Postgres is the source of truth |
