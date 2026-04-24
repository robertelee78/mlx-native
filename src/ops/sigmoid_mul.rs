//! Elementwise sigmoid-gated multiply: `out[i] = x[i] * sigmoid(gate[i])`.
//!
//! Used by Qwen3.5 full-attention's output gate (ADR-013 Decision 9).
//! Sigmoid (not swish) is the authoritative activation — HF transformers
//! `modeling_qwen3_5.py:689` and vLLM `qwen3_next.py:312-314` both apply
//! `torch.sigmoid(gate)`. The kernel matches those implementations exactly.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

pub static SIGMOID_MUL_SHADER_SOURCE: &str = include_str!("../shaders/sigmoid_mul.metal");

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("sigmoid_mul_f32", SIGMOID_MUL_SHADER_SOURCE);
    registry.register_source("sigmoid_mul_bf16", SIGMOID_MUL_SHADER_SOURCE);
}

/// Dispatch `output[i] = x[i] * sigmoid(gate[i])` elementwise.
///
/// All three buffers must share dtype and element count (= `n`). `params_buf`
/// carries a single u32 `[n]`.
pub fn dispatch_sigmoid_mul(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    x: &MlxBuffer,
    gate: &MlxBuffer,
    output: &MlxBuffer,
    params_buf: &MlxBuffer,
    n: u32,
) -> Result<()> {
    if n == 0 {
        return Err(MlxError::InvalidArgument(
            "sigmoid_mul: n must be > 0".into(),
        ));
    }
    let expected = n as usize;
    for (name, buf) in [("x", x), ("gate", gate), ("output", output)] {
        if buf.element_count() != expected {
            return Err(MlxError::InvalidArgument(format!(
                "sigmoid_mul: {} element count {} != n {}",
                name,
                buf.element_count(),
                n
            )));
        }
    }
    if x.dtype() != gate.dtype() || x.dtype() != output.dtype() {
        return Err(MlxError::InvalidArgument(format!(
            "sigmoid_mul: dtype mismatch x={}, gate={}, output={}",
            x.dtype(),
            gate.dtype(),
            output.dtype()
        )));
    }
    let kernel_name = match x.dtype() {
        DType::F32 => "sigmoid_mul_f32",
        DType::BF16 => "sigmoid_mul_bf16",
        other => {
            return Err(MlxError::InvalidArgument(format!(
                "sigmoid_mul: unsupported dtype {}",
                other
            )));
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, device)?;
    let tg = MTLSize::new(std::cmp::min(n as u64, 256), 1, 1);
    let grid = MTLSize::new(n as u64, 1, 1);

    encoder.encode(
        pipeline,
        &[(0, x), (1, gate), (2, output), (3, params_buf)],
        grid,
        tg,
    );
    Ok(())
}
