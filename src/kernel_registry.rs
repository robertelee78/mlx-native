//! [`KernelRegistry`] — lazy compilation and caching of Metal compute pipelines.
//!
//! MSL shader source is embedded at compile time via `include_str!`.  On first
//! access, the source is compiled into a Metal library, the named function is
//! extracted, and a `ComputePipelineState` is created and cached.  Subsequent
//! calls return the cached pipeline.

use std::collections::HashMap;

use metal::ComputePipelineState;

use crate::error::{MlxError, Result};

/// Registry that lazily compiles and caches Metal compute pipelines from
/// embedded MSL source.
///
/// # Usage
///
/// ```ignore
/// let mut registry = KernelRegistry::new();
/// let pipeline = registry.get_pipeline("elementwise_add", device.metal_device())?;
/// encoder.encode(&pipeline, &buffers, grid, tg);
/// ```
///
/// # Thread Safety
///
/// `KernelRegistry` is **not** `Sync` by default (it uses `&mut self` for
/// `get_pipeline` to allow mutable cache insertion).  If you need concurrent
/// access, wrap it in a `Mutex` or use one registry per thread.
pub struct KernelRegistry {
    /// Cached pipelines keyed by kernel function name.
    cache: HashMap<String, ComputePipelineState>,
    /// MSL source text keyed by kernel function name.
    ///
    /// Populated at construction time with all embedded shader sources.
    sources: HashMap<String, &'static str>,
}

impl KernelRegistry {
    /// Create a new registry with all embedded shader sources pre-registered.
    ///
    /// No compilation happens here — shaders are compiled lazily on first use.
    pub fn new() -> Self {
        let mut sources = HashMap::new();

        // Register embedded shader sources.
        // Future stories will add real kernels here, e.g.:
        //   sources.insert("elementwise_add".into(), include_str!("shaders/elementwise_add.metal"));
        //
        // For now, register the placeholder so the embed machinery is exercised.
        sources.insert(
            "placeholder".into(),
            include_str!("shaders/placeholder.metal"),
        );

        Self {
            cache: HashMap::new(),
            sources,
        }
    }

    /// Register a shader source at runtime (useful for testing and dynamic
    /// kernel generation).
    pub fn register_source(&mut self, name: impl Into<String>, source: &'static str) {
        let name = name.into();
        // Invalidate any cached pipeline for this name since the source changed.
        self.cache.remove(&name);
        self.sources.insert(name, source);
    }

    /// Get a compiled compute pipeline for the named kernel function.
    ///
    /// On first call for a given name, this compiles the MSL source into a
    /// Metal library, extracts the named function, and creates a
    /// `ComputePipelineState`.  Subsequent calls return the cached pipeline.
    ///
    /// # Errors
    ///
    /// * `MlxError::KernelNotFound` — no source registered for this name.
    /// * `MlxError::ShaderCompilationError` — MSL compilation or pipeline
    ///   creation failed.
    pub fn get_pipeline(
        &mut self,
        name: &str,
        device: &metal::DeviceRef,
    ) -> Result<&ComputePipelineState> {
        if !self.cache.contains_key(name) {
            // Slow path: compile the shader.
            let source = self.sources.get(name).ok_or_else(|| {
                MlxError::KernelNotFound(name.to_string())
            })?;

            let compile_opts = metal::CompileOptions::new();
            let library = device
                .new_library_with_source(source, &compile_opts)
                .map_err(|msg| MlxError::ShaderCompilationError {
                    name: name.to_string(),
                    message: msg,
                })?;

            let function = library
                .get_function(name, None)
                .map_err(|msg| MlxError::ShaderCompilationError {
                    name: name.to_string(),
                    message: msg,
                })?;

            let pipeline = device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|msg| MlxError::ShaderCompilationError {
                    name: name.to_string(),
                    message: msg,
                })?;

            self.cache.insert(name.to_string(), pipeline);
        }

        // At this point the pipeline is guaranteed to be in the cache.
        // We use `ok_or_else` instead of `expect` to satisfy the no-panic policy.
        self.cache.get(name).ok_or_else(|| {
            MlxError::KernelNotFound(name.to_string())
        })
    }

    /// Check if a pipeline for the given name is already compiled and cached.
    pub fn is_cached(&self, name: &str) -> bool {
        self.cache.contains_key(name)
    }

    /// Number of compiled pipelines currently in the cache.
    pub fn cached_count(&self) -> usize {
        self.cache.len()
    }

    /// Number of registered shader sources.
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }
}

impl Default for KernelRegistry {
    fn default() -> Self {
        Self::new()
    }
}
