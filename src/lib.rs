//! # mlx-native
//!
//! Pure-Rust Metal GPU compute library for MLX-compatible inference on Apple
//! Silicon.
//!
//! This crate provides a thin, safe wrapper around Apple's Metal framework
//! focused on compute shader dispatch for neural network inference.  It is
//! designed to be the GPU backend for the `hf2q` inference engine.
//!
//! ## Key Types
//!
//! | Type | Purpose |
//! |------|---------|
//! | [`MlxDevice`]       | Metal device + command queue (entry point) |
//! | [`CommandEncoder`]   | Batched compute command submission |
//! | [`MlxBuffer`]        | Typed Metal buffer with shape/dtype metadata |
//! | [`MlxBufferPool`]    | Arena allocator with power-of-two bucketing |
//! | [`KernelRegistry`]   | Lazy MSL compilation + pipeline cache |
//! | [`DType`]            | Element data type enum |
//! | [`MlxError`]         | Unified error type (never panics) |
//!
//! ## Quick Start
//!
//! ```ignore
//! use mlx_native::{MlxDevice, DType};
//!
//! let device = MlxDevice::new()?;
//! let buf = device.alloc_buffer(1024, DType::F32, vec![256])?;
//! let encoder = device.command_encoder()?;
//! ```
//!
//! ## Design Principles
//!
//! * **No panics** — all public APIs return `Result<T, MlxError>`.
//! * **Zero-copy** — `StorageModeShared` buffers on Apple Silicon unified memory.
//! * **Thread-safe** — `MlxDevice` and `MlxBuffer` are `Send + Sync`.
//! * **Lazy compilation** — MSL shaders compiled on first use, then cached.

// Enforce the no-panic policy at compile time.
#![deny(clippy::panic, clippy::unwrap_used, clippy::expect_used)]
// The `objc` crate's `msg_send!` macro internally checks `cfg(feature = "cargo-clippy")`
// which triggers unexpected_cfgs warnings. Suppress at crate level since we can't
// control the macro expansion site.
#![allow(unexpected_cfgs)]

// ---- internal modules ----
#[macro_use]
mod error;
mod buffer;
mod buffer_pool;
mod device;
mod dtypes;
mod encoder;
mod kernel_registry;
pub mod gguf;
pub mod kernel_profile;
pub mod graph;
pub mod ops;
pub mod turboquant;
pub mod weight;

// ---- public re-exports ----
pub use buffer::MlxBuffer;
pub use buffer_pool::MlxBufferPool;
pub use device::MlxDevice;
pub use dtypes::DType;
pub use encoder::{
    barrier_count, barrier_total_ns, cmd_buf_count, dispatch_count, reset_counters, sync_count,
    CapturedNode, CommandEncoder, DispatchKind, RecordedBinding,
};
pub use error::{MlxError, Result};
pub use graph::{ComputeGraph, GraphExecutor, GraphSession, OpKind};
pub use kernel_registry::KernelRegistry;

// Re-export GGUF parser.
pub use gguf::{GgufFile, MetadataValue, TensorInfo};

// Re-export ops.
pub use ops::dense_mm_bf16::{dense_matmul_bf16_f32_tensor, DenseMmBf16F32Params};
pub use ops::dense_mm_f16::{dense_matmul_f16_f32_tensor, DenseMmF16F32Params};
pub use ops::dense_mm_f32_f32::{dense_matmul_f32_f32_tensor, DenseMmF32F32Params};
pub use ops::quantized_matmul::{quantized_matmul, quantized_matmul_simd, QuantizedMatmulParams};
pub use ops::quantized_matmul_ggml::{
    dispatch_mm_for_test, quantized_matmul_ggml, quantized_matmul_mm_tensor_perm021,
    GgmlQuantizedMatmulParams, GgmlQuantizedMatmulPerm021Params, GgmlType,
    MM_ROUTING_THRESHOLD,
};
pub use ops::quantized_matmul_id::{quantized_matmul_id, QuantizedMatmulIdParams};
pub use ops::quantized_matmul_id_ggml::{
    dispatch_id_mm_for_test, quantized_matmul_id_ggml, quantized_matmul_id_ggml_pooled,
    quantized_matmul_id_swiglu_q4_0,
    GgmlIdMmDispatchParams, GgmlQuantizedMatmulIdParams, IdMmScratch,
    MM_ID_ROUTING_THRESHOLD,
};

// Re-export weight loading utilities.
pub use weight::{
    load_quantized_weights, safetensors_to_metal_buffer, QuantizationConfig, QuantizedWeight,
    SafetensorsFile, TensorQuantConfig,
};

// Re-export metal types that appear in the public API.
pub use metal::MTLSize;
pub use metal;

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    // ---- T10.7: compile-time Send + Sync assertions ----
    fn _assert_send<T: Send>() {}
    fn _assert_sync<T: Sync>() {}

    #[allow(dead_code)]
    fn assert_send_sync() {
        _assert_send::<MlxDevice>();
        _assert_sync::<MlxDevice>();
        _assert_send::<MlxBuffer>();
        _assert_sync::<MlxBuffer>();
        _assert_send::<MlxError>();
        _assert_sync::<MlxError>();
    }

    // ---- T10.1: device initialization ----
    #[test]
    fn test_device_init() {
        let device = MlxDevice::new().expect("MlxDevice::new() should succeed on Apple Silicon");
        let name = device.name();
        assert!(!name.is_empty(), "Device name should not be empty");
        println!("Metal device: {name}");
    }

    // ---- T10.2: buffer allocation ----
    #[test]
    fn test_buffer_alloc() {
        let device = MlxDevice::new().expect("device");
        let shape = vec![2, 3, 4];
        let byte_len = 2 * 3 * 4 * DType::F32.size_of(); // 96 bytes
        let buf = device
            .alloc_buffer(byte_len, DType::F32, shape.clone())
            .expect("alloc_buffer");

        assert_eq!(buf.dtype(), DType::F32);
        assert_eq!(buf.shape(), &shape);
        assert_eq!(buf.byte_len(), byte_len);
        assert_eq!(buf.element_count(), 24);
    }

    // ---- T10.3: buffer read/write round-trip ----
    #[test]
    fn test_buffer_readwrite() {
        let device = MlxDevice::new().expect("device");
        let n = 64;
        let byte_len = n * std::mem::size_of::<f32>();
        let mut buf = device
            .alloc_buffer(byte_len, DType::F32, vec![n])
            .expect("alloc_buffer");

        // Write known data.
        {
            let slice: &mut [f32] = buf.as_mut_slice().expect("as_mut_slice");
            assert_eq!(slice.len(), n);
            for (i, val) in slice.iter_mut().enumerate() {
                *val = i as f32 * 1.5;
            }
        }

        // Read back and verify.
        {
            let slice: &[f32] = buf.as_slice().expect("as_slice");
            for (i, &val) in slice.iter().enumerate() {
                let expected = i as f32 * 1.5;
                assert!(
                    (val - expected).abs() < f32::EPSILON,
                    "Mismatch at index {i}: got {val}, expected {expected}"
                );
            }
        }
    }

    // ---- T10.4: encoder lifecycle ----
    #[test]
    fn test_encoder_lifecycle() {
        let device = MlxDevice::new().expect("device");
        let mut enc = device.command_encoder().expect("command_encoder");
        // Commit an empty command buffer — should succeed (no-op on GPU).
        enc.commit_and_wait()
            .expect("commit_and_wait on empty encoder");
    }

    // ---- T10.5: buffer pool reuse ----
    #[test]
    fn test_buffer_pool_reuse() {
        let device = MlxDevice::new().expect("device");
        let mut pool = MlxBufferPool::new();

        // Allocate a buffer.
        let buf1 = pool
            .alloc(&device, 1024, DType::F32, vec![256])
            .expect("pool alloc 1");
        let buf1_ptr = buf1.contents_ptr();
        let buf1_byte_len = buf1.byte_len();

        // Release it back to the pool.
        pool.release(buf1);
        assert_eq!(pool.free_count(), 1);

        // Allocate again — should reuse the same Metal buffer.
        let buf2 = pool
            .alloc(&device, 1024, DType::F32, vec![256])
            .expect("pool alloc 2");
        let buf2_ptr = buf2.contents_ptr();
        let buf2_byte_len = buf2.byte_len();

        assert_eq!(buf1_ptr, buf2_ptr, "Pool should reuse the same Metal buffer");
        assert_eq!(buf1_byte_len, buf2_byte_len, "Byte lengths should match");
        assert_eq!(pool.free_count(), 0, "Free list should be empty after reuse");
    }

    // ---- T10.6: kernel registry caching ----
    #[test]
    fn test_kernel_registry_caching() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();

        // Register a minimal test kernel.
        registry.register_source(
            "test_add",
            r#"
            #include <metal_stdlib>
            using namespace metal;
            kernel void test_add(
                device float *a [[buffer(0)]],
                device float *b [[buffer(1)]],
                device float *c [[buffer(2)]],
                uint id [[thread_position_in_grid]]
            ) {
                c[id] = a[id] + b[id];
            }
            "#,
        );

        // First call — compiles the shader.
        assert!(!registry.is_cached("test_add"));
        let p1 = registry
            .get_pipeline("test_add", device.metal_device())
            .expect("get_pipeline first call");
        let p1_ptr = p1 as *const _;
        assert!(registry.is_cached("test_add"));

        // Second call — returns cached pipeline.
        let p2 = registry
            .get_pipeline("test_add", device.metal_device())
            .expect("get_pipeline second call");
        let p2_ptr = p2 as *const _;

        assert_eq!(
            p1_ptr, p2_ptr,
            "Second get_pipeline call should return the same cached pipeline"
        );
    }

    // ---- Additional: test alloc_buffer with zero length returns error ----
    #[test]
    fn test_buffer_alloc_zero_len_error() {
        let device = MlxDevice::new().expect("device");
        let result = device.alloc_buffer(0, DType::F32, vec![]);
        assert!(result.is_err(), "Zero-length allocation should fail");
        match result {
            Err(MlxError::InvalidArgument(_)) => {}
            other => panic!("Expected InvalidArgument, got {:?}", other),
        }
    }

    // ---- Additional: test kernel not found ----
    #[test]
    fn test_kernel_not_found() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        let result = registry.get_pipeline("nonexistent_kernel", device.metal_device());
        assert!(result.is_err());
        match result {
            Err(MlxError::KernelNotFound(name)) => {
                assert_eq!(name, "nonexistent_kernel");
            }
            other => panic!("Expected KernelNotFound, got {:?}", other),
        }
    }

    // ---- Additional: test DType properties ----
    #[test]
    fn test_dtype_sizes() {
        assert_eq!(DType::F32.size_of(), 4);
        assert_eq!(DType::F16.size_of(), 2);
        assert_eq!(DType::BF16.size_of(), 2);
        assert_eq!(DType::U8.size_of(), 1);
        assert_eq!(DType::U16.size_of(), 2);
        assert_eq!(DType::U32.size_of(), 4);
        assert_eq!(DType::I32.size_of(), 4);
    }

    // ---- Additional: test MlxBuffer Debug ----
    #[test]
    fn test_buffer_debug() {
        let device = MlxDevice::new().expect("device");
        let buf = device
            .alloc_buffer(64, DType::F16, vec![4, 8])
            .expect("alloc_buffer");
        let debug_str = format!("{:?}", buf);
        assert!(debug_str.contains("MlxBuffer"));
        assert!(debug_str.contains("F16"));
        assert!(debug_str.contains("[4, 8]"));
    }

    // ---- Additional: test MlxError Display ----
    #[test]
    fn test_error_display() {
        let e = MlxError::DeviceNotFound;
        assert!(format!("{e}").contains("Metal GPU device"));

        let e = MlxError::ShaderCompilationError {
            name: "foo".into(),
            message: "syntax error".into(),
        };
        assert!(format!("{e}").contains("foo"));
        assert!(format!("{e}").contains("syntax error"));
    }

    // ---- Additional: test buffer pool with different sizes ----
    #[test]
    fn test_buffer_pool_size_buckets() {
        let device = MlxDevice::new().expect("device");
        let mut pool = MlxBufferPool::new();

        // Allocate a 100-byte buffer (rounds to 128-byte bucket).
        let buf_100 = pool.alloc(&device, 100, DType::U8, vec![100]).expect("alloc 100");
        assert!(
            buf_100.byte_len() >= 100,
            "Buffer should be at least 100 bytes"
        );
        pool.release(buf_100);

        // Allocate a 128-byte buffer — should reuse the same Metal buffer.
        let buf_128 = pool.alloc(&device, 128, DType::U8, vec![128]).expect("alloc 128");
        assert!(buf_128.byte_len() >= 128);
        pool.release(buf_128);

        // Allocate a 200-byte buffer — different bucket (256), fresh allocation.
        let buf_200 = pool.alloc(&device, 200, DType::U8, vec![200]).expect("alloc 200");
        assert!(buf_200.byte_len() >= 200);
        pool.release(buf_200);

        assert_eq!(pool.free_count(), 2, "Two different bucket sizes in pool");
    }
}
