//! GPU kernel host-side dispatch functions.
//!
//! Each submodule implements dispatch for a specific kernel family.

pub mod elementwise;
pub mod embedding;
pub mod encode_helpers;
pub mod gelu;
pub mod moe_dispatch;
pub mod moe_gate;
pub mod quantized_matmul;
pub mod rms_norm;
pub mod rope;
pub mod sdpa;
pub mod sdpa_sliding;
pub mod softcap;
pub mod softmax;
pub mod transpose;
