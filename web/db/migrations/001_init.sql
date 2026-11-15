-- RecoSphere Play: initial schema (P1). Target: Postgres 16+ (Neon or the local compose Postgres).
-- Postgres holds USER-OWNED STATE only (accounts, library, plays, feedback, admin actions).
-- Not here, on purpose: exposure/click event streams (P3: Redpanda -> lake), model artifacts and
-- embeddings (P2: MinIO), experiment assignment (deterministic salted hash, nothing to store).
-- No BEGIN/COMMIT here: web/db/migrate.mjs runs each file in ONE transaction together with its
-- schema_migrations row. Forward-only: never edit this file once applied; add 002_*.sql.

CREATE EXTENSION IF NOT EXISTS citext;   -- case-insensitive email
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- fuzzy title search

CREATE TYPE user_kind      AS ENUM ('registered', 'demo');
CREATE TYPE user_role      AS ENUM ('player', 'admin');
CREATE TYPE item_type      AS ENUM ('game', 'dlc', 'accessory', 'console', 'other');
CREATE TYPE library_source AS ENUM ('onboarding', 'play');
CREATE TYPE feedback_kind  AS ENUM ('like', 'dislike', 'not_interested');

-- 1. users: one row per person who uses the app. Demo players (real dataset users) get a row
--    on their first demo login so that their in-app plays can be stored like anyone else's.
CREATE TABLE users (
    id               uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    kind             user_kind   NOT NULL DEFAULT 'registered',
    email            citext      UNIQUE,
    password_hash    text,                                   -- bcrypt; never the password
    display_name     text        NOT NULL,
    role             user_role   NOT NULL DEFAULT 'player',
    dataset_user_id  text        UNIQUE,                     -- Amazon reviewer id (demo players only)
    onboarded_at     timestamptz,                            -- NULL = still needs the 3-pick onboarding
    created_at       timestamptz NOT NULL DEFAULT now(),
    last_login_at    timestamptz,
    CONSTRAINT registered_has_credentials
        CHECK (kind <> 'registered' OR (email IS NOT NULL AND password_hash IS NOT NULL)),
    CONSTRAINT demo_has_dataset_id
        CHECK (kind <> 'demo' OR dataset_user_id IS NOT NULL)
);

-- 2. sessions: server-side record behind the JWT cookie (the JWT carries sid = sessions.id).
--    Makes logout, "sign out everywhere" and banning real: a stateless JWT cannot be revoked.
CREATE TABLE sessions (
    id          uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     uuid        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at  timestamptz NOT NULL DEFAULT now(),
    expires_at  timestamptz NOT NULL,
    revoked_at  timestamptz,
    user_agent  text,
    CHECK (expires_at > created_at)
);
CREATE INDEX sessions_active_by_user ON sessions (user_id) WHERE revoked_at IS NULL;

-- 3. games: the app's copy of the catalog (loaded from gold metadata by a script, upserted per
--    model release). Gives plays/library a foreign key, the game/accessory flag, dedup families
--    and search. The engine keeps its own catalog view for the model; an item is shown only if
--    it exists in both.
CREATE TABLE games (
    item_id       text        PRIMARY KEY,                   -- ASIN, the id the engine uses
    title         text        NOT NULL,
    title_family  text        NOT NULL,                      -- normalized title: dedup editions/platforms
    item_type     item_type   NOT NULL,
    platforms     text[]      NOT NULL DEFAULT '{}',
    publisher     text,
    description   text,
    listed_on     date,                                      -- Amazon "date first available", NOT release date
    is_active     boolean     NOT NULL DEFAULT true,         -- soft-remove without breaking history FKs
    source        text        NOT NULL DEFAULT 'amazon_reviews_2023',
    added_at      timestamptz NOT NULL DEFAULT now(),
    updated_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX games_family     ON games (title_family);
CREATE INDEX games_playable   ON games (item_id) WHERE item_type IN ('game', 'dlc') AND is_active;
CREATE INDEX games_title_trgm ON games USING gin (title gin_trgm_ops);

-- 4. library: current state, one row per (user, game). Onboarding picks land here too.
--    This is what the app sends to the engine as the user's history, and what "Continue playing"
--    reads. Updated in the SAME transaction as the play_sessions insert, so it never drifts.
CREATE TABLE library (
    user_id         uuid           NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    item_id         text           NOT NULL REFERENCES games(item_id),
    source          library_source NOT NULL,
    added_at        timestamptz    NOT NULL DEFAULT now(),
    last_played_at  timestamptz,
    play_count      integer        NOT NULL DEFAULT 0 CHECK (play_count >= 0),
    seconds_played  bigint         NOT NULL DEFAULT 0 CHECK (seconds_played >= 0),
    PRIMARY KEY (user_id, item_id)
);
CREATE INDEX library_recent ON library (user_id, (COALESCE(last_played_at, added_at)) DESC);

-- 5. play_sessions: append-only history of every stream. Keeps attribution (which recommendation
--    request and which row led to the play) for offline evaluation and later retraining.
CREATE TABLE play_sessions (
    id               uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id          uuid        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    item_id          text        NOT NULL REFERENCES games(item_id),
    client_event_id  uuid        NOT NULL UNIQUE,            -- idempotency key: a retried request is a no-op
    started_at       timestamptz NOT NULL DEFAULT now(),
    ended_at         timestamptz,
    seconds_played   integer     CHECK (seconds_played >= 0),
    request_id       text,                                   -- engine request that showed the card; NULL = search/direct
    surface          text,                                   -- e.g. 'home:top_picks', 'game:more_like_this', 'search'
    CHECK (ended_at IS NULL OR ended_at >= started_at)
);
CREATE INDEX play_sessions_by_user ON play_sessions (user_id, started_at DESC);
CREATE INDEX play_sessions_by_item ON play_sessions (item_id, started_at DESC);

-- 6. item_feedback: explicit signals that serving must respect ("not interested" is filtered out).
--    One current value per (user, game); changing your mind overwrites it.
CREATE TABLE item_feedback (
    user_id     uuid          NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    item_id     text          NOT NULL REFERENCES games(item_id),
    kind        feedback_kind NOT NULL,
    created_at  timestamptz   NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, item_id)
);

-- 7. admin_audit_log: who did what from /admin (model rollback, experiment stop, role change).
--    Append-only; the actor link survives user deletion as NULL.
CREATE TABLE admin_audit_log (
    id          bigint      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    actor_id    uuid        REFERENCES users(id) ON DELETE SET NULL,
    action      text        NOT NULL,                        -- 'model.rollback', 'experiment.stop', ...
    target      text,                                        -- e.g. the run id rolled back to
    details     jsonb       NOT NULL DEFAULT '{}'::jsonb,
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX admin_audit_recent ON admin_audit_log (created_at DESC);
