//! Shared fail-closed contract for native BF16 dense projections.

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::error::{MlxError, Result};
use crate::ops::dense_mm_bf16::DenseMmBf16F32Params;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DenseBf16Contract {
    RowReduction,
    MatrixTensor,
    MatrixSimdgroup,
}

fn checked_required_bytes(
    operation: &str,
    name: &str,
    dimensions: &[u32],
    element_size: usize,
) -> Result<usize> {
    dimensions
        .iter()
        .try_fold(1usize, |product, &dimension| {
            product.checked_mul(dimension as usize)
        })
        .and_then(|elements| elements.checked_mul(element_size))
        .ok_or_else(|| {
            MlxError::InvalidArgument(format!("{operation}: {name} size overflows usize"))
        })
}

pub(crate) fn validate_dense_bf16_contract(
    operation: &str,
    contract: DenseBf16Contract,
    src0: &MlxBuffer,
    src1: &MlxBuffer,
    dst: &MlxBuffer,
    params: &DenseMmBf16F32Params,
) -> Result<()> {
    if params.m == 0 || params.n == 0 || params.k == 0 {
        return Err(MlxError::InvalidArgument(format!(
            "{operation}: M, N, K must all be > 0"
        )));
    }
    match contract {
        DenseBf16Contract::RowReduction if params.k % 4 != 0 => {
            return Err(MlxError::InvalidArgument(format!(
                "{operation}: K ({}) must be divisible by 4 for aligned vector loads",
                params.k
            )));
        }
        DenseBf16Contract::MatrixTensor | DenseBf16Contract::MatrixSimdgroup if params.k < 32 => {
            return Err(MlxError::InvalidArgument(format!(
                "{operation}: K ({}) must be at least 32",
                params.k
            )));
        }
        DenseBf16Contract::MatrixTensor if params.k % 4 != 0 => {
            return Err(MlxError::InvalidArgument(format!(
                "{operation}: tensor K ({}) must be divisible by 4 so every F32 input row is 16-byte aligned",
                params.k
            )));
        }
        DenseBf16Contract::RowReduction
        | DenseBf16Contract::MatrixTensor
        | DenseBf16Contract::MatrixSimdgroup => {}
    }
    for (name, dimension) in [
        ("M", params.m),
        ("N", params.n),
        ("K", params.k),
        ("src0_batch", params.src0_batch),
        ("src1_batch", params.src1_batch),
    ] {
        if dimension > i32::MAX as u32 {
            return Err(MlxError::InvalidArgument(format!(
                "{operation}: {name} ({dimension}) exceeds i32 shader indexing"
            )));
        }
    }
    if params.src0_batch == 0 || params.src1_batch == 0 {
        return Err(MlxError::InvalidArgument(format!(
            "{operation}: batch counts must be > 0"
        )));
    }
    if params.src1_batch % params.src0_batch != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "{operation}: src1_batch ({}) must be a multiple of src0_batch ({}) for GQA broadcast",
            params.src1_batch, params.src0_batch
        )));
    }
    let broadcast = params.src1_batch / params.src0_batch;
    if broadcast > i16::MAX as u32 {
        return Err(MlxError::InvalidArgument(format!(
            "{operation}: GQA broadcast ratio ({broadcast}) exceeds i16::MAX"
        )));
    }
    if src0.dtype() != DType::BF16 || src1.dtype() != DType::F32 || dst.dtype() != DType::F32 {
        return Err(MlxError::InvalidArgument(format!(
            "{operation}: expected BF16/F32/F32 buffers, got {:?}/{:?}/{:?}",
            src0.dtype(),
            src1.dtype(),
            dst.dtype()
        )));
    }
    for (name, offset, alignment) in [
        ("src0", src0.byte_offset(), 8u64),
        ("src1", src1.byte_offset(), 16u64),
        ("dst", dst.byte_offset(), 4u64),
    ] {
        if offset % alignment != 0 {
            return Err(MlxError::InvalidArgument(format!(
                "{operation}: {name} byte offset {offset} is not {alignment}-byte aligned"
            )));
        }
    }
    let expected_src0 = checked_required_bytes(
        operation,
        "src0",
        &[params.src0_batch, params.n, params.k],
        DType::BF16.size_of(),
    )?;
    let expected_src1 = checked_required_bytes(
        operation,
        "src1",
        &[params.src1_batch, params.m, params.k],
        DType::F32.size_of(),
    )?;
    let expected_dst = checked_required_bytes(
        operation,
        "dst",
        &[params.src1_batch, params.m, params.n],
        DType::F32.size_of(),
    )?;
    for (name, actual, required) in [
        ("src0", src0.data_byte_len(), expected_src0),
        ("src1", src1.data_byte_len(), expected_src1),
        ("dst", dst.data_byte_len(), expected_dst),
    ] {
        if actual < required {
            return Err(MlxError::InvalidArgument(format!(
                "{operation}: {name} needs {required} logical bytes, got {actual}"
            )));
        }
    }
    Ok(())
}
