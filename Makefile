# RecoSphere: one entry point for the local stack. Run from the repo root in Git Bash, WSL or Linux/macOS.
#   make up       core: web app :3000, engine :8000, redis        make up-all   + obs (Prometheus, Grafana)
#   make down     stop everything (volumes are kept)              make test     unit tests
#   make smoke    whole system on synthetic data (no Docker)      make config   validate compose for every profile
# Config: .env (created from .env.example on first use). Profiles: see infra/compose/docker-compose.yml.

COMPOSE  := docker compose -f infra/compose/docker-compose.yml --env-file .env
PROFILES := obs
PY       ?= python

.DEFAULT_GOAL := help
.PHONY: help up up-all down logs ps test smoke config

help:            ## list the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | sed -E 's/:.*## /\t/'

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

config:          ## validate the compose file for core and for every profile (no daemon needed)
	docker compose -f infra/compose/docker-compose.yml --env-file .env.example config --quiet
	@for p in $(PROFILES); do \
	  docker compose -f infra/compose/docker-compose.yml --env-file .env.example --profile $$p config --quiet || exit 1; \
	done
	@echo "compose config ok: core + $(PROFILES)"
