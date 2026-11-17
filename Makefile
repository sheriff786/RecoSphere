# RecoSphere: one entry point for the local stack. Run from the repo root in Git Bash, WSL or Linux/macOS.
#   make up       core: web app :3000, engine :8000, redis        make up-all   + obs (Prometheus, Grafana)
#   make down     stop everything (volumes are kept)              make test     unit tests
#   make smoke    whole system on synthetic data (no Docker)      make config   validate compose for every profile
#   make e2e      Playwright against the running stack            make e2e-synthetic   same on a throwaway stack (CI)
# Config: .env (created from .env.example on first use). Profiles: see infra/compose/docker-compose.yml.

COMPOSE  := docker compose -f infra/compose/docker-compose.yml --env-file .env
PROFILES := obs
PY       ?= python

.DEFAULT_GOAL := help
.PHONY: help up up-all down logs ps test smoke config e2e e2e-synthetic db-migrate db-status db-check app-api db-upgrade db-current db-drift

help:            ## list the targets
	@grep -E '^[a-z0-9-]+:.*## ' $(MAKEFILE_LIST) | sed -E 's/:.*## /\t/'

.env:            # first run: start from the template (never overwrites an existing .env)
	cp .env.example .env
	@echo "created .env from .env.example; change the secrets in it before exposing the stack"

up: .env         ## core stack: web, engine, redis (builds images if needed)
	$(COMPOSE) up --build -d
	@echo "app http://localhost:3000 | engine http://localhost:8000/health (first engine start loads models, ~1-2 min) | make ps"

up-all: .env     ## core + every optional profile (obs)
	$(COMPOSE) $(foreach p,$(PROFILES),--profile $(p)) up --build -d

down: .env       ## stop and remove all containers, every profile (named volumes are kept)
	$(COMPOSE) $(foreach p,$(PROFILES),--profile $(p)) down

logs: .env       ## follow logs of the running services
	$(COMPOSE) $(foreach p,$(PROFILES),--profile $(p)) logs -f --tail 100

ps: .env         ## running services and health
	$(COMPOSE) $(foreach p,$(PROFILES),--profile $(p)) ps

test:            ## unit tests
	$(PY) -m pytest -q

smoke:           ## end-to-end smoke test on synthetic data (pipeline -> gate -> engine -> parity -> A/B)
	bash scripts/ci_smoke.sh

e2e:             ## browser e2e (Playwright) against the running stack: make up first. Never rolls back.
	cd web && npx playwright test

e2e-synthetic:   ## browser e2e on a throwaway synthetic stack, incl. roll back (what CI runs; no Docker)
	bash scripts/e2e_stack.sh

# app_api (P1, docs/platform/P1_ROADMAP.md). URLs from app_api/.env (git-ignored). Schema changes: models +
# Alembic (`$(PY) -m alembic revision --autogenerate -m "..."`), never by hand.
app-api:         ## run app_api (user data on Postgres) on :8100
	$(PY) -m uvicorn app_api.main:app --host 127.0.0.1 --port 8100

db-upgrade:      ## apply pending Alembic migrations to the app database
	$(PY) -m alembic upgrade head

db-current:      ## show the app database's Alembic revision
	$(PY) -m alembic current

db-drift:        ## fail if the models and the app database differ (Alembic autogenerate finds changes)
	$(PY) -m alembic check

# Legacy Node runner (frozen since P1 phase A; removed in phase H). URLs from web/.env.local.
db-migrate:      ## apply pending SQL migrations (web/db/migrations) to the app database
	cd web && node --env-file=.env.local db/migrate.mjs

db-status:       ## list applied / pending migrations, change nothing
	cd web && node --env-file=.env.local db/migrate.mjs --status

db-check:        ## run every schema rule against the app database (one transaction, rolled back)
	cd web && node --env-file=.env.local db/check_schema.mjs

config:          ## validate the compose file for core and for every profile (no daemon needed)
	docker compose -f infra/compose/docker-compose.yml --env-file .env.example config --quiet
	@for p in $(PROFILES); do \
	  docker compose -f infra/compose/docker-compose.yml --env-file .env.example --profile $$p config --quiet || exit 1; \
	done
	@echo "compose config ok: core + $(PROFILES)"
