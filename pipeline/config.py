"""Config loading: YAML defaults + `--set a.b=value` overrides + paths."""
import copy
import os
from pathlib import Path

import yaml

VARIANTS = {
    "id_only":        dict(use_id=True,  use_content=False, use_age=False),
    "content_only":   dict(use_id=False, use_content=True,  use_age=False),
    "id_content":     dict(use_id=True,  use_content=True,  use_age=False),
    "id_content_age": dict(use_id=True,  use_content=True,  use_age=True),
}
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


def load_config(path=None, overrides=()):
    cfg = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text())
    for ov in overrides:                                     # "two_tower.epochs=2"
        key, _, raw = ov.partition("=")
        node = cfg
        *parents, leaf = key.strip().split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)
    return cfg


class Paths:
    def __init__(self, cfg):
        self.category = cfg["category"]
        self.data_root = Path(os.environ.get("RANKFORGE_DATA", "data")).resolve()
        self.repo_root = Path(os.environ.get("RANKFORGE_REPO", ".")).resolve()
        self.gold = self.data_root / "gold" / self.category
        self.models = self.repo_root / "models" / self.category
        if cfg.get("namespace"):                             # e.g. "replay/q2021": isolated runs + pointer
            self.models = self.models / cfg["namespace"]
        self.runs = self.models / "runs"
        self.production_pointer = self.models / "production.json"
        self.production_history = self.models / "production_history.jsonl"


def two_tower_cfg(cfg) -> dict:
    """Checkpoint cfg dict in the exact shape notebook 03 / retrieval.two_tower expect."""
    t = copy.deepcopy(cfg["two_tower"])
    return {k: t[k] for k in ("seed", "text_encoder", "hist_len", "dim", "tau", "batch_size", "lr",
                              "weight_decay", "id_dropout")} | {"epochs": t["epochs"]}
