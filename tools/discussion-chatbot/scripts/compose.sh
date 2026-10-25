#!/usr/bin/env bash
set -euo pipefail
app_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export LOCAL_UID="${LOCAL_UID:-$(id -u)}"
export LOCAL_GID="${LOCAL_GID:-$(id -g)}"
export RESEARCH_REVISION="$(git -C "$app_dir/../.." rev-parse HEAD 2>/dev/null || printf unknown)"
exec docker compose --project-directory "$app_dir" -f "$app_dir/compose.yaml" "$@"
