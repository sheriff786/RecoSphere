"""docs/API.md must match the real API surface, per service: section "1. Engine API" = the engine's routes,
"2. Web app API" = web/app/api/**/route.ts, "3. App API" = app_api's routes. In each section every route has a
table row, no row points at a route that no longer exists, and (for the two FastAPI services) every endpoint's
"### METHOD `path`" part names all of its query and body parameters. Sections are compared separately because
the same path can exist in two services (both FastAPI services have GET /health)."""
import re
from pathlib import Path

import pytest

from app_api.main import app as app_api
from serve.app import app as engine

ROOT = Path(__file__).resolve().parents[1]
DOC = (ROOT / "docs" / "API.md").read_text(encoding="utf-8")
ROW = re.compile(r"^\| (GET|POST|PUT|PATCH|DELETE) \| `(/[^`]*)` \|", re.M)
HANDLER = re.compile(r"export\s+(?:async\s+)?function\s+(GET|POST|PUT|PATCH|DELETE)\b")


def part(n: int) -> str:
    """Text of section '## n. ...' up to the next '## ' heading."""
    m = re.search(rf"^## {n}\. .*$", DOC, re.M)
    assert m, f"docs/API.md has no '## {n}.' section"
    nxt = DOC.find("\n## ", m.end())
    return DOC[m.start(): nxt if nxt > 0 else len(DOC)]


def openapi_routes(fastapi_app, at_least: int) -> set[tuple[str, str]]:
    # From the OpenAPI spec (= what Swagger shows). app.routes is not reliable: FastAPI >= 0.16x keeps
    # included routers as lazy _IncludedRouter objects there.
    out = {(m.upper(), p) for p, ops in fastapi_app.openapi()["paths"].items() for m in ops}
    assert len(out) >= at_least, f"routes not found ({out}); the test is broken, not the doc"
    return out


def web_routes() -> set[tuple[str, str]]:
    api = ROOT / "web" / "app" / "api"
    out = set()
    for f in api.rglob("route.ts"):
        path = "/api/" + f.parent.relative_to(api).as_posix()
        out |= {(m, path) for m in HANDLER.findall(f.read_text(encoding="utf-8"))}
    assert len(out) >= 5, "web routes not found; the test is broken, not the doc"
    return out


SERVICES = {
    "engine": (1, lambda: openapi_routes(engine, 10)),
    "web": (2, web_routes),
    "app_api": (3, lambda: openapi_routes(app_api, 1)),
}


@pytest.mark.parametrize("service", SERVICES)
def test_every_route_is_documented(service):
    n, routes = SERVICES[service]
    missing = routes() - set(ROW.findall(part(n)))
    assert not missing, f"add these to section {n} of docs/API.md (table row + section): {sorted(missing)}"


@pytest.mark.parametrize("service", SERVICES)
def test_no_documented_route_is_gone(service):
    n, routes = SERVICES[service]
    stale = set(ROW.findall(part(n))) - routes()
    assert not stale, f"section {n} of docs/API.md documents routes that no longer exist: {sorted(stale)}"


def endpoint_text(text: str, method: str, path: str) -> str:
    start = text.find(f"### {method} `{path}`")
    assert start >= 0, f"docs/API.md has no '### {method} `{path}`' in this section"
    nxt = [i for i in (text.find("\n### ", start + 1), text.find("\n## ", start + 1)) if i > 0]
    return text[start: min(nxt) if nxt else len(text)]


@pytest.mark.parametrize("service", ["engine", "app_api"])
def test_sections_name_every_parameter(service):
    fastapi_app, n = (engine, 1) if service == "engine" else (app_api, 3)
    spec, text = fastapi_app.openapi(), part(n)
    schemas = spec.get("components", {}).get("schemas", {})
    problems = []
    for path, ops in spec["paths"].items():
        for method, op in ops.items():
            names = [p["name"] for p in op.get("parameters", []) if p["in"] == "query"]
            ref = (op.get("requestBody", {}).get("content", {}).get("application/json", {})
                   .get("schema", {}).get("$ref"))
            if ref:
                names += list(schemas[ref.split("/")[-1]]["properties"])
            body = endpoint_text(text, method.upper(), path)
            problems += [f"{method.upper()} {path}: `{x}`" for x in names if f"`{x}`" not in body and f'"{x}"' not in body]
    assert not problems, f"parameters missing from their docs/API.md section: {problems}"
