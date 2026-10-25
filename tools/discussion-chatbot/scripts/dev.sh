#!/usr/bin/env bash
set -euo pipefail
app_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
backend_pid=''
cleanup() { if [[ -n "$backend_pid" ]]; then kill "$backend_pid" 2>/dev/null || true; wait "$backend_pid" 2>/dev/null || true; fi; }
trap cleanup EXIT INT TERM
cd "$app_dir/backend"
[[ -d node_modules ]] || npm ci
npm run build
HOST=127.0.0.1 PORT=8787 node dist/server.js &
backend_pid=$!
cd "$app_dir/frontend"
[[ -d node_modules ]] || npm ci
npm run dev -- --host 127.0.0.1
