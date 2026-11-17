"""RecoSphere app API: user-owned data (accounts, sessions, library, plays, feedback, admin audit) on Postgres.

Classic ORM layout, each layer only calls the one below it:
    routers/  (HTTP)  ->  services/  (business rules, transactions)  ->  models/  (SQLAlchemy classes)
    ->  db/  (engine, pool, sessions)  ->  Postgres (Neon)
schemas/ holds the Pydantic request/response models used by routers and services.
Plan and phases: docs/platform/P1_ROADMAP.md. Tables: DB_SCHEMA.md.
"""
