"""
One-off repair: a replay ran with an old pipeline/config.py that ignored `namespace`, so its runs
(cut-*) landed in models/<cat>/runs/ and were PROMOTED in the REAL production pointer.

This moves everything the replay wrote into models/<cat>/replay/<name>/ and restores the real
production pointer to the last non-replay model in the history. Nothing is deleted.

    python scripts/fix_replay_leak.py --name q90            # dry run: only prints what it would do
    python scripts/fix_replay_leak.py --name q90 --apply
"""
import argparse
import json
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="replay name used with pipeline.replay --name")
    ap.add_argument("--category", default="Video_Games")
    ap.add_argument("--prefix", default="cut-", help="run ids the replay created")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    root = Path("models") / a.category
    dest = root / "replay" / a.name
    hist_path, ptr = root / "production_history.jsonl", root / "production.json"
    hist = [json.loads(x) for x in hist_path.read_text().splitlines() if x.strip()]
    leaked = [h for h in hist if h["run_id"].startswith(a.prefix)]
    kept = [h for h in hist if not h["run_id"].startswith(a.prefix)]
    runs = sorted(p for p in (root / "runs").glob(f"{a.prefix}*") if p.is_dir())
    reports = [root / f for f in ("timeline.md", "timeline.json", "decay.md", "decay.csv", "decay.png", "policies.md")
               if (root / f).exists()]
    cur = json.loads(ptr.read_text()) if ptr.exists() else None
    restore = kept[-1] if kept else None

    print(f"current production : {cur['run_id'] if cur else None}")
    print(f"restore production : {restore['run_id'] if restore else None}  (last non-replay entry in history)")
    print(f"move {len(runs)} run folders  -> {dest / 'runs'}: {[r.name for r in runs]}")
    print(f"move {len(reports)} report files -> {dest}: {[r.name for r in reports]}")
    print(f"history: keep {len(kept)} lines, move {len(leaked)} replay lines to {dest / 'production_history.jsonl'}")
    if not a.apply:
        print("\ndry run - nothing changed. Re-run with --apply.")
        return
    if restore is None:
        raise SystemExit("no non-replay model in the history to restore - fix production.json by hand")

    (dest / "runs").mkdir(parents=True, exist_ok=True)
    for r in runs:
        shutil.move(str(r), str(dest / "runs" / r.name))
    for f in reports:
        shutil.move(str(f), str(dest / f.name))
    # paths in history are "runs/<id>" relative to the models root -> still valid under replay/<name>/
    (dest / "production_history.jsonl").write_text("".join(json.dumps(h) + "\n" for h in leaked))
    if leaked:
        (dest / "production.json").write_text(json.dumps(leaked[-1], indent=2))
    hist_path.write_text("".join(json.dumps(h) + "\n" for h in kept))
    tmp = ptr.with_suffix(".tmp")
    tmp.write_text(json.dumps(restore, indent=2))
    tmp.replace(ptr)
    print(f"\ndone. production -> {restore['run_id']}. A running API hot-swaps back within MODEL_WATCH_SECONDS.")


if __name__ == "__main__":
    main()
