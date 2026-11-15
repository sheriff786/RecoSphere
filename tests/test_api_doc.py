"""docs/API.md must match the real API surface: every engine route and every web /api/* route has a table
row, no row points at a route that no longer exists, and each engine endpoint's section names all of its
query and body parameters. Fails CI when an endpoint is added, changed or removed without a doc update."""
import re
from pathlib import Path

from serve.app import app

ROOT = Path(__file__).resolve().parents[1]
DOC = (ROOT / "docs" / "API.md").read_text(encoding="utf-8")
ROW = re.compile(r"^\| (GET|POST|PUT|PATCH|DELETE) \| `(/[^`]*)` \|", re.M)
HANDLER = re.compile(r"export\s+(?:async\s+)?function\s+(GET|POST|PUT|PATCH|DELETE)\b")


def engine_routes() -> set[tuple[str, str]]:
    # From the OpenAPI spec (= what Swagger shows). app.routes is not reliable: FastAPI >= 0.16x keeps
    # included routers as lazy _IncludedRouter objects there.
    out = {(m.upper(), p) for p, ops in app.openapi()["paths"].items() for m in ops}
    assert len(out) >= 10, f"engine routes not found ({out}); test is broken, not the doc"
    return out


def web_routes() -> set[tuple[str, str]]:
    api = ROOT / "web" / "app" / "api"
    out = set()
    for f in api.rglob("route.ts"):
        path = "/api/" + f.parent.relative_to(api).as_posix()
        out |= {(m, path) for m in HANDLER.findall(f.read_text(encoding="utf-8"))}
    return out


def documented() -> set[tuple[str, str]]:
    return set(ROW.findall(DOC))


def test_every_route_is_documented():
    missing = (engine_routes() | web_routes()) - documented()
    assert not missing, f"add these to docs/API.md (table row + section): {sorted(missing)}"


def test_no_documented_route_is_gone():
    stale = documented() - (engine_routes() | web_routes())
    assert not stale, f"docs/API.md documents routes that no longer exist: {sorted(stale)}"


def section(method: str, path: str) -> str:
    start = DOC.find(f"### {method} `{path}`")
    assert start >= 0, f"docs/API.md has no section '### {method} `{path}`'"
    nxt = [i for i in (DOC.find("\n### ", start + 1), DOC.find("\n## ", start + 1)) if i > 0]
    return DOC[start: min(nxt) if nxt else len(DOC)]


def test_engine_sections_name_every_parameter():
    spec = app.openapi()
    schemas = spec.get("components", {}).get("schemas", {})
    problems = []
    for path, ops in spec["paths"].items():
        for method, op in ops.items():
            names = [p["name"] for p in op.get("parameters", []) if p["in"] == "query"]
            ref = (op.get("requestBody", {}).get("content", {}).get("application/json", {})
                   .get("schema", {}).get("$ref"))
            if ref:
                names += list(schemas[ref.split("/")[-1]]["properties"])
            text = section(method.upper(), path)
            problems += [f"{method.upper()} {path}: `{n}`" for n in names if f"`{n}`" not in text and f'"{n}"' not in text]
    assert not problems, f"parameters missing from their docs/API.md section: {problems}"
