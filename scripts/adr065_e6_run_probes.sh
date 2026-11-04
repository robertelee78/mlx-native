#!/usr/bin/env bash
# ADR-065 E6 verify-window probes on both metallib paths.
#
# Usage: scripts/adr065_e6_run_probes.sh EVIDENCE_DIR
#
# Runs GPU work: the caller must hold the GPU lane and a quiet host.
# Writes E6_RECORD files, logs and provenance into EVIDENCE_DIR.
set -euo pipefail

evidence=${1:?usage: $0 EVIDENCE_DIR}
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
mkdir -p "$evidence"

for name in HF2Q_TQ_NWG HF2Q_TQ_NSG MLX_PRECOMPILED_METALLIB_FCV E6_STRICT_NAN; do
    if [ -n "${!name:-}" ]; then
        echo "refusing to run: $name is set" >&2
        exit 2
    fi
done

export RUSTC_WRAPPER= CARGO_INCREMENTAL=0
GIT_COMMIT_SHA=$(git rev-parse HEAD)
export GIT_COMMIT_SHA

correctness=(
    --test adr065_e6_mul_mv_ext_width
    --test adr065_e6_served_kv_attention_windows
    --test adr065_e6_gated_delta_net_decode_seqs
    --test adr065_e6_capture_restore_copy
    --test adr065_p4a_capture_restore
)
bandwidth=(--test adr065_e6_weight_load_bandwidth)

cargo test --release "${correctness[@]}" "${bandwidth[@]}" --no-run 2>&1 |
    tee "$evidence/build.log"

{
    echo "source_commit=$GIT_COMMIT_SHA"
    echo "dirty=$(git status --porcelain | wc -l | tr -d ' ')"
    echo "crate_version=$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)"
    echo "rustc=$(rustc --version)"
    target=${CARGO_TARGET_DIR:-$root/target}
    find "$target/release/build" -path '*mlx-native-*/out/default.metallib' \
        -exec shasum -a 256 {} + | sed 's/^/metallib_sha256=/'
    grep -oE 'Executable tests/adr065_(e6|p4a)[^ ]* \([^)]*\)' "$evidence/build.log" |
        sed -E 's/.*\((.*)\)/\1/' | while read -r binary; do
            echo "test_binary_sha256=$(shasum -a 256 "$binary")"
        done
} >"$evidence/provenance.txt"
rm -f "$evidence/status.txt"

for path in precompiled runtime; do
    if [ "$path" = runtime ]; then flag=0; else flag=1; fi
    rm -f "$evidence/records-$path.txt" "$evidence/bandwidth-$path.txt"
    status=0
    MLX_PRECOMPILED_METALLIB=$flag E6_RECORD_PATH="$evidence/records-$path.txt" \
        cargo test --release "${correctness[@]}" --no-fail-fast -- --test-threads=1 2>&1 |
        tee "$evidence/test-$path.log" || status=$?
    echo "correctness_$path=$status" >>"$evidence/status.txt"
    status=0
    MLX_PRECOMPILED_METALLIB=$flag E6_STRICT_NAN=1 \
        E6_RECORD_PATH="$evidence/strict-nan-$path.txt" \
        cargo test --release --test adr065_e6_capture_restore_copy -- --test-threads=1 2>&1 |
        tee "$evidence/strict-nan-$path.log" || status=$?
    echo "strict_nan_$path=$status" >>"$evidence/status.txt"
    # Set E6_SKIP_BANDWIDTH=1 to skip the timing probe (no quiet host).
    if [ "${E6_SKIP_BANDWIDTH:-0}" = 1 ]; then continue; fi
    status=0
    MLX_PRECOMPILED_METALLIB=$flag E6_RECORD_PATH="$evidence/bandwidth-$path.txt" \
        cargo test --release "${bandwidth[@]}" -- --ignored --test-threads=1 2>&1 |
        tee "$evidence/bandwidth-$path.log" || status=$?
    echo "bandwidth_$path=$status" >>"$evidence/status.txt"
done
