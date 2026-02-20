#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RENDER_DEPLOY_HOOK_URL:-}" ]]; then
  echo "RENDER_DEPLOY_HOOK_URL is missing."
  echo "Set it first, then run again."
  exit 1
fi

echo "Triggering Render deploy hook..."
curl --fail --show-error --silent --request POST "$RENDER_DEPLOY_HOOK_URL"
echo "Render deploy triggered."
