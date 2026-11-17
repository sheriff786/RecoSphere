"""A throwaway Postgres database for one e2e run (scripts/e2e_stack.sh). Never the app's real database.

    python scripts/e2e_db.py create <server_url>          -> prints the new database's URL (recosphere_e2e_<hex>)
    python scripts/e2e_db.py drop   <server_url> <db_url>  -> drops it again (only recosphere_e2e_* names)

<server_url> = any database on the server, used to run CREATE/DROP DATABASE (CI: the postgres service; laptop:
Neon's direct host). Works with sslmode=verify-full (certifi's CA bundle, like app_api).
"""
import secrets
import sys
from urllib.parse import urlsplit, urlunsplit

import certifi
import psycopg

PREFIX = "recosphere_e2e_"


def _connect(url: str):
    kw = {"sslrootcert": certifi.where()} if "sslmode=verify" in url else {}
    return psycopg.connect(url, autocommit=True, **kw)


def _with_db(url: str, name: str) -> str:
    p = urlsplit(url)
    return urlunsplit((p.scheme, p.netloc, "/" + name, p.query, p.fragment))


def create(server_url: str) -> str:
    name = PREFIX + secrets.token_hex(4)
    with _connect(server_url) as c:
        c.execute(f'CREATE DATABASE "{name}"')
    return _with_db(server_url, name)


def drop(server_url: str, db_url: str) -> None:
    name = urlsplit(db_url).path.lstrip("/")
    if not name.startswith(PREFIX):
        raise SystemExit(f"refusing to drop {name!r}: not an e2e database")
    with _connect(server_url) as c:
        c.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "create":
        print(create(sys.argv[2]))
    elif cmd == "drop":
        drop(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(__doc__)
