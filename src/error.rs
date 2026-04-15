//! Error types for the mlx-native crate.
//!
//! All public functions return `Result<T, MlxError>` — the crate never panics.

/// Compile-time assertion that a type is Send + Sync.  Used internally.
#[doc(hidden)]
#[macro_export]
macro_rules! static_assertions_send_sync {
    ($t:ty) => {
        const _: fn() = || {
            fn assert_send<T: Send>() {}
            fn assert_sync<T: Sync>() {}
            assert_send::<$t>();
            assert_sync::<$t>();
        };
    };
}

/// Unified error type for all Metal GPU operations.
#[derive(Debug, thiserror::Error)]
pub enum MlxError {
    /// No Metal-capable GPU device was found on this system.
    #[error("No Metal GPU device found — Apple Silicon required")]
    DeviceNotFound,

    /// A Metal command buffer completed with an error status.
    #[error("Command buffer error: {0}")]
    CommandBufferError(String),

    /// An MSL shader failed to compile.
    #[error("Shader compilation error for '{name}': {message}")]
    ShaderCompilationError {
        /// Name of the shader / kernel function that failed.
        name: String,
        /// Compiler diagnostic message.
        message: String,
    },

    /// Metal buffer allocation failed (usually out of GPU memory).
    #[error("Failed to allocate Metal buffer of {bytes} bytes")]
    BufferAllocationError {
        /// Requested allocation size in bytes.
        bytes: usize,
    },

    /// An argument to a public function was invalid.
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    /// A kernel function was not found in the compiled library.
    #[error("Kernel not found: {0}")]
    KernelNotFound(String),

    /// An I/O error occurred (e.g. reading a safetensors file).
    #[error("I/O error: {0}")]
    IoError(String),

    /// A safetensors file could not be parsed or contains invalid data.
    #[error("Safetensors error: {0}")]
    SafetensorsError(String),

    /// A quantization config file could not be parsed.
    #[error("Quantization config error: {0}")]
    QuantConfigError(String),

    /// An unsupported data type was encountered.
    #[error("Unsupported dtype: {0}")]
    UnsupportedDtype(String),

    /// A GGUF file could not be parsed or contains invalid data.
    #[error("GGUF parse error: {0}")]
    GgufParseError(String),
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, MlxError>;

/// Display implementation is handled by thiserror; this is a manual `Debug`
/// helper for the shader variant to keep log output readable.
impl MlxError {
    /// Returns `true` if this is a transient error that *might* succeed on retry
    /// (e.g. a command buffer timeout). Most errors are permanent.
    pub fn is_transient(&self) -> bool {
        matches!(self, MlxError::CommandBufferError(_))
    }
}

// Ensure the error type itself is thread-safe.
static_assertions_send_sync!(MlxError);
