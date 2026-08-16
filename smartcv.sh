#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "$here/core/core.py" ]]; then
  root="$here"
elif [[ -f "$here/core.py" ]]; then
  root="$(cd "$here/.." && pwd)"
else
  echo "SmartCV: cannot find project root (core/core.py)." >&2
  exit 1
fi
cd "$root"

if command -v python3 >/dev/null 2>&1; then
  py=python3
elif command -v python >/dev/null 2>&1; then
  py=python
else
  echo "SmartCV: python not found. Install Python 3.12+." >&2
  exit 1
fi

set +e
"$py" -m core.core
status=$?
set -e
echo
read -r -p "Press Enter to close..."
exit "$status"
