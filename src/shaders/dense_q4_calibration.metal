#include <metal_stdlib>

using namespace metal;

constant uint Q4_PROOF_POISON = 0xffffffffu;

struct DenseQ4ProofAuxParams {
    ulong logical_elements;
    ulong guarded_elements;
    uint status_index;
};

enum DenseQ4ProofStatus : uint {
    DenseQ4ProofControlUnwritten = 1u << 0,
    DenseQ4ProofControlNonFinite = 1u << 1,
    DenseQ4ProofCandidateUnwritten = 1u << 2,
    DenseQ4ProofCandidateNonFinite = 1u << 3,
    DenseQ4ProofMismatch = 1u << 4,
    DenseQ4ProofControlGuard = 1u << 5,
    DenseQ4ProofCandidateGuard = 1u << 6,
};

kernel void hf2q_dense_q4_proof_poison(
    device uint *control [[buffer(0)]],
    device uint *candidate [[buffer(1)]],
    constant DenseQ4ProofAuxParams &params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
    if ((ulong)tid >= params.guarded_elements) {
        return;
    }
    control[tid] = Q4_PROOF_POISON;
    candidate[tid] = Q4_PROOF_POISON;
}

kernel void hf2q_dense_q4_proof_compare(
    device const uint *control [[buffer(0)]],
    device const uint *candidate [[buffer(1)]],
    device atomic_uint *statuses [[buffer(2)]],
    constant DenseQ4ProofAuxParams &params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
    if ((ulong)tid >= params.guarded_elements) {
        return;
    }

    const uint control_bits = control[tid];
    const uint candidate_bits = candidate[tid];
    uint status = 0;
    if ((ulong)tid < params.logical_elements) {
        if (control_bits == Q4_PROOF_POISON) {
            status |= DenseQ4ProofControlUnwritten;
        } else if ((control_bits & 0x7f800000u) == 0x7f800000u) {
            status |= DenseQ4ProofControlNonFinite;
        }
        if (candidate_bits == Q4_PROOF_POISON) {
            status |= DenseQ4ProofCandidateUnwritten;
        } else if ((candidate_bits & 0x7f800000u) == 0x7f800000u) {
            status |= DenseQ4ProofCandidateNonFinite;
        }
        if (control_bits != candidate_bits) {
            status |= DenseQ4ProofMismatch;
        }
    } else {
        if (control_bits != Q4_PROOF_POISON) {
            status |= DenseQ4ProofControlGuard;
        }
        if (candidate_bits != Q4_PROOF_POISON) {
            status |= DenseQ4ProofCandidateGuard;
        }
    }
    if (status != 0) {
        atomic_fetch_or_explicit(
            &statuses[params.status_index], status, memory_order_relaxed);
    }
}
