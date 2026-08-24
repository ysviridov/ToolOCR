#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INCOMING="${1:-$ROOT/data/incoming}"

latest="$(find "$INCOMING" -maxdepth 1 -type f -name 'ADDRESS_*.zip' -printf '%p\n' | sort | tail -n 1)"
if [[ -z "${latest:-}" ]]; then
  echo "В $INCOMING не найдено файлов ADDRESS_*.zip" >&2
  exit 2
fi

make -C "$ROOT" update FILE="$latest"
