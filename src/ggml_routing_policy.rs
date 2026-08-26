//! Canonical process-environment resolution for GGML dispatch policy.
//!
//! Capability receipts and production dispatch must use the same resolved
//! values.  Keeping one public resolver prevents callers from reconstructing
//! only the dense or expert half of the policy and accidentally certifying a
//! different route from the one the legacy environment-backed entrypoints
//! execute.

use crate::ggml_capability::GgmlRoutingPolicy;
use crate::kernel_registry::KernelRegistry;
use crate::ops::quantized_matmul_ggml::dense_routing_policy_from_environment;
use crate::ops::quantized_matmul_id_ggml::expert_routing_policy_from_environment;

/// Resolve every GGML routing knob from the process environment exactly once.
///
/// The underlying hot-path flags retain their existing cached semantics.  A
/// caller that needs a reproducible execution receipt should resolve this at
/// model load, serialize it, and pass the same value to the explicit
/// `*_with_policy` entrypoints and [`crate::ggml_capability`].
pub fn ggml_routing_policy_from_environment() -> GgmlRoutingPolicy {
    let dense = dense_routing_policy_from_environment();
    let expert = expert_routing_policy_from_environment();
    combine_routing_policies(dense, expert)
}

/// Resolve the policy for an execution registry. Model owners freeze a policy
/// before readiness; standalone and legacy registries retain environment
/// compatibility until explicitly bound.
pub(crate) fn ggml_routing_policy_for_registry(registry: &KernelRegistry) -> GgmlRoutingPolicy {
    registry
        .ggml_routing_policy()
        .copied()
        .unwrap_or_else(ggml_routing_policy_from_environment)
}

fn combine_routing_policies(
    dense: GgmlRoutingPolicy,
    expert: GgmlRoutingPolicy,
) -> GgmlRoutingPolicy {
    GgmlRoutingPolicy {
        dense_q5k_canonical_q4x4: dense.dense_q5k_canonical_q4x4,
        dense_decode_mvn: dense.dense_decode_mvn,
        dense_decode_mv_ext: dense.dense_decode_mv_ext,
        dense_q6k_mv_nr2: dense.dense_q6k_mv_nr2,
        dense_q8_0_mv_nr2: dense.dense_q8_0_mv_nr2,
        dense_tensor_mm: dense.dense_tensor_mm,
        allow_dense_large_tile_mm: dense.allow_dense_large_tile_mm,
        expert_mm_threshold: expert.expert_mm_threshold,
        expert_q6k_mv_nr2: expert.expert_q6k_mv_nr2,
        expert_q8_0_mv_nr2: expert.expert_q8_0_mv_nr2,
        expert_tensor_mm: expert.expert_tensor_mm,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml_capability::GgmlTensorMmPreference;

    #[test]
    fn canonical_resolver_combines_dense_and_expert_halves() {
        let dense = GgmlRoutingPolicy {
            dense_q5k_canonical_q4x4: true,
            dense_decode_mvn: false,
            dense_decode_mv_ext: true,
            dense_q6k_mv_nr2: false,
            dense_q8_0_mv_nr2: false,
            dense_tensor_mm: GgmlTensorMmPreference::ForceSimd,
            allow_dense_large_tile_mm: false,
            ..GgmlRoutingPolicy::default()
        };
        let expert = GgmlRoutingPolicy {
            expert_mm_threshold: 77,
            expert_q6k_mv_nr2: false,
            expert_q8_0_mv_nr2: true,
            expert_tensor_mm: GgmlTensorMmPreference::ForceSimd,
            ..GgmlRoutingPolicy::default()
        };
        let policy = combine_routing_policies(dense, expert);
        assert!(policy.dense_q5k_canonical_q4x4);
        assert!(!policy.dense_decode_mvn);
        assert!(policy.dense_decode_mv_ext);
        assert!(!policy.dense_q6k_mv_nr2);
        assert!(!policy.dense_q8_0_mv_nr2);
        assert_eq!(policy.dense_tensor_mm, GgmlTensorMmPreference::ForceSimd);
        assert!(!policy.allow_dense_large_tile_mm);
        assert_eq!(policy.expert_mm_threshold, 77);
        assert!(!policy.expert_q6k_mv_nr2);
        assert!(policy.expert_q8_0_mv_nr2);
        assert_eq!(policy.expert_tensor_mm, GgmlTensorMmPreference::ForceSimd);
    }

    #[test]
    fn environment_override_helper() {
        if std::env::var_os("MLX_NATIVE_ROUTING_POLICY_TEST_CHILD").is_none() {
            return;
        }
        let policy = ggml_routing_policy_from_environment();
        let expected_q5 =
            std::env::var("MLX_NATIVE_ROUTING_POLICY_EXPECT_Q5").as_deref() == Ok("1");
        assert_eq!(policy.dense_q5k_canonical_q4x4, expected_q5);
        assert!(!policy.dense_decode_mvn);
        assert!(policy.dense_decode_mv_ext);
        assert!(!policy.dense_q6k_mv_nr2);
        assert!(!policy.dense_q8_0_mv_nr2);
        assert_eq!(policy.dense_tensor_mm, GgmlTensorMmPreference::ForceSimd);
        assert!(!policy.allow_dense_large_tile_mm);
        assert_eq!(policy.expert_mm_threshold, 77);
        assert!(!policy.expert_q6k_mv_nr2);
        assert!(policy.expert_q8_0_mv_nr2);
        assert_eq!(policy.expert_tensor_mm, GgmlTensorMmPreference::ForceSimd);
    }

    #[test]
    fn public_resolver_matches_process_overrides() {
        let run = |q5: Option<&str>, expected_q5: &str| {
            let mut command =
                std::process::Command::new(std::env::current_exe().expect("current test exe"));
            command
                .arg("--exact")
                .arg("ggml_routing_policy::tests::environment_override_helper")
                .arg("--nocapture")
                .env("MLX_NATIVE_ROUTING_POLICY_TEST_CHILD", "1")
                .env("MLX_NATIVE_ROUTING_POLICY_EXPECT_Q5", expected_q5)
                .env("HF2Q_DECODE_MVN", "0")
                .env("HF2Q_DECODE_MV_EXT", "1")
                .env("HF2Q_Q6K_MV_NR2", "0")
                .env("HF2Q_Q8_0_MV_NR2", "0")
                .env("HF2Q_DISABLE_TENSOR_MM", "1")
                .env("HF2Q_LARGE_TILE_MM", "0")
                .env("HF2Q_MM_ID_ROUTING_THRESHOLD", "77")
                .env("HF2Q_Q6K_ID_MV_NR2", "0")
                .env("HF2Q_Q8_0_ID_MV_NR2", "1")
                .env("HF2Q_DISABLE_TENSOR_MM_ID", "1");
            if let Some(value) = q5 {
                command.env("HF2Q_Q5K_CANONICAL_Q4X4", value);
            } else {
                command.env_remove("HF2Q_Q5K_CANONICAL_Q4X4");
            }
            command
                .output()
                .expect("run isolated routing-policy helper")
        };
        for (value, expected) in [(None, "1"), (Some("1"), "1"), (Some("0"), "0")] {
            let output = run(value, expected);
            assert!(
                output.status.success(),
                "routing-policy child failed for {value:?}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(
                String::from_utf8_lossy(&output.stdout).contains("running 1 test"),
                "routing-policy child executed no exact test for {value:?}: {}",
                String::from_utf8_lossy(&output.stdout)
            );
        }
    }
}
