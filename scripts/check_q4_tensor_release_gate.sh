#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ -n "${MLX_NATIVE_SKIP_METALLIB+x}" ]]; then
  echo "release gate refuses MLX_NATIVE_SKIP_METALLIB, even when it is empty" >&2
  exit 2
fi

export MLX_NATIVE_REQUIRE_METAL_TENSOR_ARTIFACT=1

gate_logs="$(mktemp -d "${TMPDIR:-/tmp}/mlx-native-q4-release.XXXXXX")"
trap 'rm -rf -- "$gate_logs"' EXIT

run_nonzero_test_target() {
  local target="$1"
  local log="$gate_logs/$target.log"

  cargo test --locked --release --test "$target" 2>&1 | tee "$log"
  local result_count
  local nonzero_ok_count
  result_count="$(grep -Ec '^test result: ' "$log" || true)"
  nonzero_ok_count="$(grep -Ec '^test result: ok\. [1-9][0-9]* passed;' "$log" || true)"
  if [[ "$result_count" -ne 1 || "$nonzero_ok_count" -ne 1 ]]; then
    echo "release gate failed closed: $target must report exactly one nonzero passing test result" >&2
    exit 1
  fi
}

run_nonzero_test_target dense_q4_auto_calibration
run_nonzero_test_target q4_mm_tensor_64x32
run_nonzero_test_target q4_benchmark_contract
