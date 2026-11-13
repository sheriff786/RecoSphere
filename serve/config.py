"""Serving config — everything comes from environment variables (Docker-friendly).

Which model is served: models/<cat>/production.json (written by `python -m pipeline.run`, Phase 7).
No pointer -> the flat notebook folder models/<cat>/ (Phase 6 behaviour). MODEL_DIR overrides both.
`models` is re-read on every access, so a promotion/rollback is visible without a restart (Phase 7d).
"""
import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServeConfig:
    category: str
    data_root: Path
    repo_root: Path
    redis_url: str | None
    new_item_slots: int
    threads: int
    force_reload: bool
    watch_seconds: float       # poll production.json this often and hot-swap on change (0 = off)
    admin_token: str | None    # if set, POST /admin/reload needs header X-Admin-Token

    @classmethod
    def from_env(cls) -> "ServeConfig":
        return cls(
            category=os.environ.get("RANKFORGE_CATEGORY", "Video_Games"),
            data_root=Path(os.environ.get("RANKFORGE_DATA", "data")).resolve(),
            repo_root=Path(os.environ.get("RANKFORGE_REPO", ".")).resolve(),
            redis_url=os.environ.get("REDIS_URL") or None,           # unset = in-memory store (dev)
            new_item_slots=int(os.environ.get("NEW_ITEM_SLOTS", 2)),  # notebook 06 decision
            threads=int(os.environ.get("SERVE_THREADS", 1)),          # 1 thread/request = stable p95
            force_reload=os.environ.get("FORCE_RELOAD", "0") == "1",
            watch_seconds=float(os.environ.get("MODEL_WATCH_SECONDS", 10)),
            admin_token=os.environ.get("ADMIN_TOKEN") or None,
        )

    @property
    def gold(self) -> Path:
        return self.data_root / "gold" / self.category

    @property
    def models_root(self) -> Path:
        return self.repo_root / "models" / self.category

    @property
    def production(self) -> dict | None:
        p = self.models_root / "production.json"
        return json.loads(p.read_text()) if p.exists() else None

    @property
    def models(self) -> Path:
        if os.environ.get("MODEL_DIR"):
            return Path(os.environ["MODEL_DIR"]).resolve()
        prod = self.production
        if prod is None:
            return self.models_root
        p = Path(prod["path"])
        return p if p.is_absolute() else self.models_root / p
