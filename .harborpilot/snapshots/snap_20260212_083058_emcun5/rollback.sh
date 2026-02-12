#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
if [ ! -f "$DIR/patch.diff" ]; then
  echo "patch.diff missing" >&2
  exit 1
fi
git apply -R --3way "$DIR/patch.diff" || git apply -R "$DIR/patch.diff"
echo "Rollback applied."
