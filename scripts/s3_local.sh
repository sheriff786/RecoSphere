#!/usr/bin/env bash
# Local S3 server for the model registry (P2): SeaweedFS `weed mini` on 127.0.0.1:8333, data in .local/ (git-ignored).
#   make s3-up        (or: bash scripts/s3_local.sh)        stop with Ctrl+C
# Credentials and bucket come from .env (S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY, S3_BUCKET; see .env.example).
# Binary: SeaweedFS 4.52 from https://github.com/seaweedfs/seaweedfs/releases/tag/4.52 (windows_amd64.zip or
# linux_amd64.tar.gz), unpacked into .local/seaweedfs/4.52/. Check the .md5 published next to the download.
set -euo pipefail
cd "$(dirname "$0")/.."
VERSION=4.52
ROOT=.local/seaweedfs
BIN=$ROOT/$VERSION/weed.exe; [ -x "$BIN" ] || BIN=$ROOT/$VERSION/weed
[ -x "$BIN" ] || { echo "SeaweedFS $VERSION not found in $ROOT/$VERSION (see the header of this script)"; exit 1; }

# only the S3_* lines of .env (the file also holds other secrets this script has no business reading)
if [ -f .env ]; then eval "$(grep -E '^S3_[A-Z_]+=' .env | sed 's/^/export /')"; fi
: "${S3_ACCESS_KEY_ID:?set S3_ACCESS_KEY_ID in .env}" "${S3_SECRET_ACCESS_KEY:?set S3_SECRET_ACCESS_KEY in .env}"
BUCKET=${S3_BUCKET:-recosphere}

mkdir -p "$ROOT/data"
CONF=$ROOT/s3.json                                    # identities file; written each start, never committed
python - "$CONF" <<'PY'
import json, os, sys
json.dump({"identities": [{"name": "recosphere",
                           "credentials": [{"accessKey": os.environ["S3_ACCESS_KEY_ID"],
                                            "secretKey": os.environ["S3_SECRET_ACCESS_KEY"]}],
                           "actions": ["Admin", "Read", "List", "Tagging", "Write"]}]}, open(sys.argv[1], "w"))
PY
echo "SeaweedFS $VERSION: S3 on http://127.0.0.1:8333, bucket '$BUCKET', data $ROOT/data"
# 127.0.0.1 only (not the LAN), no admin UI (it has no password by default)
exec "$BIN" mini -dir="$ROOT/data" -ip=127.0.0.1 -ip.bind=127.0.0.1 -s3.config="$CONF" -bucket="$BUCKET" \
  -admin.ui=false
