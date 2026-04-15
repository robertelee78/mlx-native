//! GPU kernel host-side dispatch functions.
//!
//! Each submodule implements dispatch for a specific kernel family.

pub mod argmax;
pub mod argsort;
pub mod copy;
pub mod dense_gemm;
pub mod elementwise;
pub mod embedding;
pub mod gather;
pub mod gather_bench;
pub mod hadamard;
pub mod hadamard_quantize_kv;
pub mod encode_helpers;
pub mod fused_head_norm_rope;
pub mod fused_norm_add;
pub mod fused_residual_norm;
pub mod gelu;
pub mod kv_cache_copy;
pub mod moe_dispatch;
pub mod moe_gate;
pub mod quantized_matmul;
pub mod quantized_matmul_ggml;
pub mod quantized_matmul_id;
pub mod quantized_matmul_id_ggml;
pub mod rms_norm;
pub mod rope;
pub mod flash_attn_vec;
pub mod flash_attn_vec_tq;
pub mod fwht_standalone;
pub mod sdpa;
pub mod sdpa_sliding;
pub mod softcap;
pub mod softmax;
pub mod softmax_sample;
pub mod transpose;
