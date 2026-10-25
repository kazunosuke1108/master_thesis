#!/usr/bin/env bash
set -euo pipefail
app_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CODEX_HOME="$app_dir/.local/codex"
mkdir -p "$CODEX_HOME"
chmod 700 "$CODEX_HOME"
exec codex -c 'cli_auth_credentials_store="file"' login --device-auth
