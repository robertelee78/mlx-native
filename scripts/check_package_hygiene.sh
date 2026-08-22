#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

classify_forbidden() {
    awk '
      /(^|\/)\.agents(\/|$)/ ||
      /(^|\/)\.claude(\/|$)/ ||
      /(^|\/)\.codex(\/|$)/ ||
      /(^|\/)\.mcp\.json$/ ||
      /(^|\/)\.env($|\.)/ ||
      /(^|\/)ruvector\.db($|\.)/ ||
      /\.rvf($|\.)/ ||
      /\.bak$/ { print }
    '
}

classify_tracked_forbidden() {
    awk '
      /(^|\/)\.mcp\.json$/ ||
      /(^|\/)\.env($|\.)/ ||
      /(^|\/)ruvector\.db($|\.)/ ||
      /\.rvf($|\.)/ { print }
    '
}

canary=$(printf '%s\n' \
    src/lib.rs \
    agentdb.rvf \
    nested/agentdb.rvf.lock \
    .env.local \
    tests/fixture.bin | classify_forbidden)
expected_canary=$(printf '%s\n' \
    agentdb.rvf \
    nested/agentdb.rvf.lock \
    .env.local)
if [[ "$canary" != "$expected_canary" ]]; then
    echo "package hygiene classifier canary failed" >&2
    exit 1
fi

package_list=$(cargo package --locked --allow-dirty --list)
forbidden=$(printf '%s\n' "$package_list" | classify_forbidden)

if [[ -n "$forbidden" ]]; then
    echo "package contains local agent, secret, memory, or backup files:" >&2
    printf '%s\n' "$forbidden" >&2
    exit 1
fi

tracked_forbidden=$(git ls-files | classify_tracked_forbidden)
if [[ -n "$tracked_forbidden" ]]; then
    echo "repository tracks local agent, secret, memory, or backup files:" >&2
    printf '%s\n' "$tracked_forbidden" >&2
    exit 1
fi

echo "package hygiene OK"
