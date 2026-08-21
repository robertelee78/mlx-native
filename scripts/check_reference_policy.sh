#!/usr/bin/env bash
# Reference-policy gate (see AGENTS.md): the peer engine may be named only
# in the sanctioned homes below. Everywhere else, code and docs say "the
# reference implementation" and comparison data goes in
# docs/peer-benchmarks.md.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

pattern='llama[._ -]?cpp|ggerganov'
fail=0

# Sanctioned homes with a per-file cap on matching lines.
check_capped() { # <path> <max-lines>
  local path=$1 cap=$2 n
  n=$(git grep -icE "$pattern" -- "$path" | cut -d: -f2 || true)
  n=${n:-0}
  if [ "$n" -gt "$cap" ]; then
    echo "POLICY: $path has $n reference-name mentions (max $cap):" >&2
    git grep -inE "$pattern" -- "$path" >&2 || true
    fail=1
  fi
}

check_capped README.md 1
check_capped AGENTS.md 1

# Zero mentions anywhere else in tracked files. Unlimited homes and
# tooling directories are excluded.
hits=$(git grep -inE "$pattern" -- \
  ':(exclude)README.md' \
  ':(exclude)AGENTS.md' \
  ':(exclude)docs/peer-benchmarks.md' \
  ':(exclude)LICENSE-MIT-llamacpp' \
  ':(exclude)scripts/check_reference_policy.sh' \
  ':(exclude).claude' \
  ':(exclude)_bmad*' \
  || true)
if [ -n "$hits" ]; then
  echo "POLICY: peer engine referenced outside sanctioned homes (see AGENTS.md):" >&2
  echo "$hits" >&2
  echo 'Fix: say "the reference implementation" / "the peer"; move comparison data to docs/peer-benchmarks.md.' >&2
  fail=1
fi

if [ "$fail" -ne 0 ]; then
  exit 1
fi
echo "reference policy OK"
