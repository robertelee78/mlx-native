//! [`GraphExecutor`] — batched Metal dispatch for single-encoder forward passes.
//!
//! llama.cpp's speed advantage over candle is NOT the kernels (Phase 0 proved
//! candle's are as fast or faster per-call).  It is the dispatch pattern:
//! 1 encoder per command buffer instead of ~120.  This module implements that
//! pattern.
//!
//! # Usage
//!
//! ```ignore
//! let mut executor = GraphExecutor::new(device.clone());
//! let mut session = executor.begin()?;
//!
//! // All ops encode into the same command buffer — no per-op encoder creation.
//! session.rms_norm(&mut registry, device.metal_device(), input, weight, output, params, rows, dim)?;
//! session.quantized_matmul(&mut registry, &device, input, weight, scales, biases, &qparams)?;
//! session.elementwise_add(&mut registry, device.metal_device(), a, b, out, n, DType::F32)?;
//!
//! // Single GPU sync point for the entire forward pass.
//! session.finish()?;
//! ```
//!
//! # Design
//!
//! The `GraphSession` holds a single `CommandEncoder`.  Each op method delegates
//! to the existing op dispatch functions in [`crate::ops`], passing the session's
//! shared encoder.  No new Metal code is needed — the ops already work with a
//! shared encoder.  The executor just prevents creating a new encoder per op.
//!
//! # Phase 4e.1 — Graph IR
//!
//! The `ComputeGraph` type captures dispatches into a `Vec<CapturedNode>` for
//! later replay.  `GraphExecutor::begin_recorded()` starts a session in capture
//! mode: all op calls are intercepted at the `CommandEncoder` level and recorded
//! instead of being sent to Metal.  `GraphSession::finish()` detects capture
//! mode, extracts the recorded graph, and replays it into a fresh encoder via
//! `ComputeGraph::encode_sequential()`.
//!
//! The existing direct-dispatch path (`begin()`) is completely unchanged.

use metal::foreign_types::ForeignType;

use crate::device::MlxDevice;
use crate::encoder::{CapturedNode, CapturedOpKind, CommandEncoder, MemRange, RecordedBinding};
use crate::error::Result;
use crate::kernel_registry::KernelRegistry;
use crate::ops;

// Re-export types used in the public API so callers don't need separate imports.
pub use crate::buffer::MlxBuffer;
pub use crate::dtypes::DType;

// ---------------------------------------------------------------------------
// OpKind — operation classification for the reorder safety whitelist (4e.3)
// ---------------------------------------------------------------------------

/// Classification of a compute operation for reorder safety analysis.
///
/// Operations marked as reorderable can be freely reordered by the graph
/// optimizer (Phase 4e.3) as long as their data dependencies allow it.
/// Non-reorderable operations have side effects or dependencies that
/// require them to stay in their original sequential position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpKind {
    /// Matrix multiplication (reorderable).
    MatMul,
    /// Expert-routed matrix multiplication (reorderable).
    MatMulId,
    /// Normalization — RMS norm, layer norm (reorderable).
    Norm,
    /// Rotary position embedding (reorderable).
    Rope,
    /// Elementwise ops — add, mul, scale, gelu, softcap, etc. (reorderable).
    Elementwise,
    /// Memory copy — KV cache copy, embedding gather (reorderable).
    Copy,
    /// Gather/scatter (reorderable).
    Gather,
    /// Scaled dot-product attention (NOT reorderable).
    Sdpa,
    /// Softmax (NOT reorderable).
    Softmax,
    /// MoE gate with CPU readback dependency (NOT reorderable).
    MoeGate,
    /// Anything else (NOT reorderable).
    Other,
}

impl OpKind {
    /// Whether this op kind is safe to reorder in the graph optimizer.
    pub fn is_reorderable(&self) -> bool {
        matches!(
            self,
            Self::MatMul
                | Self::MatMulId
                | Self::Norm
                | Self::Rope
                | Self::Elementwise
                | Self::Copy
                | Self::Gather
        )
    }
}

// ---------------------------------------------------------------------------
// ComputeGraph — the recorded graph IR
// ---------------------------------------------------------------------------

/// A recorded sequence of GPU compute dispatches and barriers.
///
/// Created by running a forward pass with the encoder in capture mode.
/// Can be replayed into a real `CommandEncoder` via `encode_sequential()`,
/// producing identical Metal dispatch behavior to the original direct path.
///
/// Future phases (4e.2, 4e.3) will add fusion and reorder passes that
/// transform the graph before encoding.
pub struct ComputeGraph {
    nodes: Vec<CapturedNode>,
}

impl ComputeGraph {
    /// Create an empty compute graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::with_capacity(128),
        }
    }

    /// Create a compute graph from a pre-built list of captured nodes.
    pub fn from_nodes(nodes: Vec<CapturedNode>) -> Self {
        Self { nodes }
    }

    /// Record a captured node into the graph.
    pub fn record(&mut self, node: CapturedNode) {
        self.nodes.push(node);
    }

    /// Number of nodes (dispatches + barriers) in the graph.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the graph contains no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Number of dispatch nodes (excludes barriers).
    pub fn dispatch_count(&self) -> usize {
        self.nodes
            .iter()
            .filter(|n| matches!(n, CapturedNode::Dispatch { .. }))
            .count()
    }

    /// Number of barrier nodes.
    pub fn barrier_count(&self) -> usize {
        self.nodes
            .iter()
            .filter(|n| matches!(n, CapturedNode::Barrier))
            .count()
    }

    /// Borrow the node list.
    pub fn nodes(&self) -> &[CapturedNode] {
        &self.nodes
    }

    /// Count dispatch nodes that have empty read/write range annotations.
    ///
    /// Used for diagnostics: if >0, the reorder pass cannot guarantee
    /// correctness because it relies on complete annotations.
    pub fn unannotated_dispatch_count(&self) -> usize {
        self.nodes
            .iter()
            .filter(|n| matches!(n, CapturedNode::Dispatch { reads, writes, .. }
                if reads.is_empty() || writes.is_empty()))
            .count()
    }

    /// Take ownership of the node list, consuming the graph.
    pub fn into_nodes(self) -> Vec<CapturedNode> {
        self.nodes
    }

    /// Encode all nodes sequentially into the given encoder.
    ///
    /// Barrier sentinel nodes emit a Metal memory barrier.  Dispatch nodes
    /// are replayed through `CommandEncoder::replay_dispatch()`.
    ///
    /// This produces identical GPU behavior to the direct-dispatch path —
    /// same pipeline bindings, same dispatch dimensions, same barrier
    /// placement.
    ///
    /// Returns the number of barriers emitted.
    pub fn encode_sequential(&self, encoder: &mut CommandEncoder) -> u32 {
        let mut barrier_count = 0u32;
        for node in &self.nodes {
            match node {
                CapturedNode::Barrier => {
                    encoder.memory_barrier();
                    barrier_count += 1;
                }
                CapturedNode::Dispatch {
                    pipeline,
                    bindings,
                    threads_per_grid,
                    threads_per_threadgroup,
                    threadgroup_memory,
                    dispatch_kind,
                    ..
                } => {
                    encoder.replay_dispatch(
                        pipeline,
                        bindings,
                        threadgroup_memory,
                        *threads_per_grid,
                        *threads_per_threadgroup,
                        *dispatch_kind,
                    );
                }
            }
        }
        barrier_count
    }

    /// Encode the graph into a Metal command buffer, computing barriers on the
    /// fly from each node's read/write buffer ranges.
    ///
    /// This is the correct encoding method for reordered graphs where barrier
    /// sentinels have been stripped.  Mirrors llama.cpp's encode-time barrier
    /// insertion via `ggml_metal_op_concurrency_check`.
    ///
    /// Returns the number of barriers emitted.
    pub fn encode_with_barriers(&self, encoder: &mut CommandEncoder) -> u32 {
        let mut tracker = ReorderConflictTracker::new();
        let mut barrier_count = 0u32;

        for node in &self.nodes {
            match node {
                CapturedNode::Dispatch {
                    pipeline,
                    bindings,
                    threads_per_grid,
                    threads_per_threadgroup,
                    threadgroup_memory,
                    dispatch_kind,
                    reads,
                    writes,
                    ..
                } => {
                    let has_ranges = !reads.is_empty() || !writes.is_empty();
                    if has_ranges && tracker.conflicts(reads, writes) {
                        encoder.memory_barrier();
                        tracker.reset();
                        barrier_count += 1;
                    }
                    if has_ranges {
                        tracker.add(reads, writes);
                    }
                    encoder.replay_dispatch(
                        pipeline,
                        bindings,
                        threadgroup_memory,
                        *threads_per_grid,
                        *threads_per_threadgroup,
                        *dispatch_kind,
                    );
                }
                CapturedNode::Barrier => {
                    // Explicit barriers still force a barrier boundary
                    encoder.memory_barrier();
                    tracker.reset();
                    barrier_count += 1;
                }
            }
        }
        barrier_count
    }

    /// Encode the graph using two command buffers for CPU/GPU overlap.
    ///
    /// The first `n0` dispatches are encoded into `encoder0` and committed
    /// immediately (GPU starts executing).  The remaining dispatches are encoded
    /// into `encoder1`.  The caller is responsible for committing `encoder1`.
    ///
    /// This matches llama.cpp's dual command buffer pattern from
    /// `ggml_metal_graph_compute` (ggml-metal-context.m:441-644):
    /// `n_nodes_0 = MAX(64, 0.1 * n_nodes)` for the first buffer.
    ///
    /// Command buffers submitted to the same `MTLCommandQueue` execute in
    /// submission order, so `encoder0.commit()` followed by `encoder1.commit()`
    /// guarantees enc0 finishes before enc1 starts.  The win: the GPU starts
    /// executing enc0 while the CPU is still encoding enc1.
    ///
    /// Returns `(barriers_buf0, barriers_buf1)`.
    pub fn encode_dual_buffer(
        &self,
        encoder0: &mut CommandEncoder,
        encoder1: &mut CommandEncoder,
    ) -> (u32, u32) {
        let dispatch_total = self.dispatch_count();
        let n0 = std::cmp::max(64, dispatch_total / 10);

        // Find the split point: the index of the n0-th dispatch node.
        let split_idx = find_dispatch_split_index(&self.nodes, n0);

        // Encode first chunk with barrier recomputation, then commit immediately.
        let barriers0 = encode_chunk_with_barriers(&self.nodes[..split_idx], encoder0);
        encoder0.commit();

        // Encode second chunk with barrier recomputation.
        let barriers1 = encode_chunk_with_barriers(&self.nodes[split_idx..], encoder1);

        (barriers0, barriers1)
    }

    /// Run the RMS norm + MUL fusion pass over the graph.
    ///
    /// Scans for the pattern:
    ///   Dispatch(RmsNorm) → Barrier(s) → Dispatch(ElemMul)
    /// where the MUL reads the norm's output buffer, and replaces the
    /// sequence with a single fused `rms_norm_mul_*` dispatch.
    ///
    /// The fused dispatch:
    /// - Reads the norm's input (buffer 0) and weight (buffer 1)
    /// - Reads the MUL's second operand as the scale (buffer 2)
    /// - Writes to the MUL's output (buffer 3)
    /// - Carries the norm's params (buffer 4)
    /// - Uses the norm's threadgroup config and shared memory
    ///
    /// Returns the number of fusions applied.
    ///
    /// # Arguments
    ///
    /// * `registry` - Kernel registry for compiling the fused pipeline.
    /// * `device`   - Metal device for pipeline compilation.
    pub fn fuse(
        &mut self,
        registry: &mut KernelRegistry,
        device: &metal::DeviceRef,
    ) -> Result<u32> {
        let mut result: Vec<CapturedNode> = Vec::with_capacity(self.nodes.len());
        let mut fusions = 0u32;
        let mut i = 0;

        while i < self.nodes.len() {
            // Check if current node is an RMS norm dispatch.
            let is_rms_norm = matches!(
                &self.nodes[i],
                CapturedNode::Dispatch { op_kind: CapturedOpKind::RmsNorm, .. }
            );

            if !is_rms_norm {
                result.push(self.nodes[i].clone());
                i += 1;
                continue;
            }

            // Look ahead: skip barriers, then check for ElemMul.
            let mut j = i + 1;
            let mut barrier_count = 0usize;
            while j < self.nodes.len() && matches!(&self.nodes[j], CapturedNode::Barrier) {
                barrier_count += 1;
                j += 1;
            }

            // Must have at least one barrier and the next node must be ElemMul.
            if barrier_count == 0 || j >= self.nodes.len() {
                result.push(self.nodes[i].clone());
                i += 1;
                continue;
            }

            let is_elem_mul = matches!(
                &self.nodes[j],
                CapturedNode::Dispatch { op_kind: CapturedOpKind::ElemMul, .. }
            );

            if !is_elem_mul {
                result.push(self.nodes[i].clone());
                i += 1;
                continue;
            }

            // Extract norm and mul dispatch fields.
            let (norm_pipeline, norm_bindings, norm_tpg, norm_tptg, norm_tgmem, norm_dk) =
                match &self.nodes[i] {
                    CapturedNode::Dispatch {
                        pipeline,
                        bindings,
                        threads_per_grid,
                        threads_per_threadgroup,
                        threadgroup_memory,
                        dispatch_kind,
                        ..
                    } => (pipeline, bindings, threads_per_grid, threads_per_threadgroup, threadgroup_memory, dispatch_kind),
                    _ => unreachable!(),
                };

            let (mul_bindings, _mul_tpg, _mul_tptg) = match &self.nodes[j] {
                CapturedNode::Dispatch {
                    bindings,
                    threads_per_grid,
                    threads_per_threadgroup,
                    ..
                } => (bindings, threads_per_grid, threads_per_threadgroup),
                _ => unreachable!(),
            };

            // Verify data dependency: the norm's output buffer (slot 2) must
            // appear as one of the MUL's input buffers (slot 0 or 1).
            //
            // Norm binding layout: (0=input, 1=weight, 2=output, 3=params)
            // MUL binding layout:  (0=a, 1=b, 2=output, 3=params_bytes)
            let norm_output_ptr = Self::buffer_ptr_for_slot(norm_bindings, 2);
            let mul_a_ptr = Self::buffer_ptr_for_slot(mul_bindings, 0);
            let mul_b_ptr = Self::buffer_ptr_for_slot(mul_bindings, 1);

            if norm_output_ptr.is_none() || (norm_output_ptr != mul_a_ptr && norm_output_ptr != mul_b_ptr) {
                // Data dependency not confirmed — don't fuse.
                result.push(self.nodes[i].clone());
                i += 1;
                continue;
            }

            // Determine which MUL input is the scale (the one that is NOT
            // the norm's output).
            let scale_slot = if norm_output_ptr == mul_a_ptr { 1 } else { 0 };

            // Build fused bindings:
            //   0 = norm input
            //   1 = norm weight
            //   2 = scale (from MUL)
            //   3 = MUL output
            //   4 = norm params
            // Gather all required bindings; bail if any are missing.
            let (norm_input, norm_weight, scale, mul_output, norm_params) = match (
                Self::get_binding(norm_bindings, 0),
                Self::get_binding(norm_bindings, 1),
                Self::get_binding(mul_bindings, scale_slot),
                Self::get_binding(mul_bindings, 2),
                Self::get_binding(norm_bindings, 3),
            ) {
                (Some(a), Some(b), Some(c), Some(d), Some(e)) => (a, b, c, d, e),
                _ => {
                    // Missing bindings — don't fuse.
                    result.push(self.nodes[i].clone());
                    i += 1;
                    continue;
                }
            };

            // Select fused pipeline based on the original norm pipeline name.
            // The norm pipeline name is "rms_norm_f32", "rms_norm_f16", or
            // "rms_norm_bf16" — we need the corresponding fused pipeline.
            let fused_name = match Self::fused_pipeline_name(norm_pipeline) {
                Some(name) => name,
                None => {
                    result.push(self.nodes[i].clone());
                    i += 1;
                    continue;
                }
            };

            let fused_pipeline = registry.get_pipeline(fused_name, device)?;

            let fused_bindings = vec![
                (0, norm_input),
                (1, norm_weight),
                (2, scale),
                (3, mul_output),
                (4, norm_params),
            ];

            // Merge read/write ranges from both the norm and mul nodes for the
            // fused dispatch.  The fused op reads everything the norm reads
            // plus the mul's scale input, and writes to the mul's output.
            let (fused_reads, fused_writes) = match (&self.nodes[i], &self.nodes[j]) {
                (
                    CapturedNode::Dispatch { reads: nr, writes: _nw, .. },
                    CapturedNode::Dispatch { reads: mr, writes: mw, .. },
                ) => {
                    let mut reads = nr.clone();
                    reads.extend_from_slice(mr);
                    (reads, mw.clone())
                }
                _ => (Vec::new(), Vec::new()),
            };

            result.push(CapturedNode::Dispatch {
                pipeline: fused_pipeline.to_owned(),
                bindings: fused_bindings,
                threads_per_grid: *norm_tpg,
                threads_per_threadgroup: *norm_tptg,
                threadgroup_memory: norm_tgmem.clone(),
                dispatch_kind: *norm_dk,
                op_kind: CapturedOpKind::Other, // Fused ops are not further fuseable
                reads: fused_reads,
                writes: fused_writes,
            });

            fusions += 1;
            // Skip past the norm, barrier(s), and mul nodes.
            i = j + 1;
        }

        self.nodes = result;
        Ok(fusions)
    }

    /// Run the reorder pass over the graph to improve GPU concurrency.
    ///
    /// Port of llama.cpp's `ggml_metal_graph_optimize_reorder` — a greedy
    /// 64-node lookahead that pulls independent dispatches forward to fill
    /// larger concurrent groups between barriers.
    ///
    /// **Prerequisites:** Call `fuse()` first if desired.  The reorder pass
    /// operates on the post-fusion graph.  Barrier sentinel nodes are stripped
    /// before reordering (they will be recomputed at encode time by the
    /// `ConflictTracker` in `encode_sequential`).
    ///
    /// **Algorithm (matching llama.cpp exactly):**
    /// 1. Strip all `CapturedNode::Barrier` nodes.
    /// 2. For each unprocessed node `i0`:
    ///    - If it conflicts with the current concurrent group (`mrs0`):
    ///      * Initialize `mrs1` from `i0`'s ranges (skipped-over set)
    ///      * Lookahead up to 64 nodes for candidates that:
    ///        (a) Are reorderable (`CapturedOpKind::is_reorderable()`)
    ///        (b) Don't conflict with `mrs0` (current group)
    ///        (c) Don't conflict with `mrs1` (skipped-over nodes)
    ///      * Pull qualifying candidates into the current group
    ///      * Non-reorderable ops break the lookahead
    ///    - Reset `mrs0` (new concurrent group)
    ///    - Add `i0` to the new group
    ///
    /// Returns the number of nodes that were moved to earlier positions.
    pub fn reorder(&mut self) -> u32 {
        // Step 1: Strip barrier nodes.  After fusion + reorder, barriers will
        // be recomputed by the ConflictTracker at encode time.
        self.nodes.retain(|n| !matches!(n, CapturedNode::Barrier));

        let n = self.nodes.len();
        if n == 0 {
            return 0;
        }

        let mut result: Vec<usize> = Vec::with_capacity(n);
        let mut used = vec![false; n];

        // mrs0: memory ranges for the current concurrent group
        let mut mrs0 = ReorderConflictTracker::new();
        // mrs1: memory ranges for skipped-over (unprocessed) nodes
        let mut mrs1 = ReorderConflictTracker::new();

        const N_FORWARD: usize = 64;

        for i0 in 0..n {
            if used[i0] {
                continue;
            }

            let node0 = &self.nodes[i0];

            // Extract reads/writes for conflict check.
            let (reads0, writes0, op_kind0) = match node0 {
                CapturedNode::Dispatch { reads, writes, op_kind, .. } => {
                    (reads.as_slice(), writes.as_slice(), *op_kind)
                }
                CapturedNode::Barrier => continue, // stripped, but be safe
            };

            // Check if node0 conflicts with the current concurrent group.
            // Empty nodes (no ranges) never conflict — like llama.cpp's is_empty.
            let has_ranges = !reads0.is_empty() || !writes0.is_empty();
            if has_ranges && mrs0.conflicts(reads0, writes0) {
                // Before starting a new group, look forward for nodes that
                // can be pulled into the CURRENT group.
                mrs1.reset();
                mrs1.add(reads0, writes0);

                let end = (i0 + N_FORWARD).min(n);
                for i1 in (i0 + 1)..end {
                    if used[i1] {
                        continue;
                    }

                    let node1 = &self.nodes[i1];
                    let (reads1, writes1, op_kind1) = match node1 {
                        CapturedNode::Dispatch { reads, writes, op_kind, .. } => {
                            (reads.as_slice(), writes.as_slice(), *op_kind)
                        }
                        CapturedNode::Barrier => continue,
                    };

                    // Non-reorderable ops break the lookahead.
                    if !op_kind1.is_reorderable() {
                        break;
                    }

                    let is_empty1 = reads1.is_empty() && writes1.is_empty();

                    // A node can be reordered into the current group if:
                    // 1. It's empty (no ranges) OR doesn't conflict with mrs0
                    // 2. It doesn't conflict with mrs1 (skipped-over nodes)
                    if (is_empty1 || !mrs0.conflicts(reads1, writes1))
                        && !mrs1.conflicts(reads1, writes1)
                    {
                        // Pull into current concurrent group.
                        mrs0.add(reads1, writes1);
                        result.push(i1);
                        used[i1] = true;
                    } else {
                        // Not eligible — expand the skipped-over set.
                        mrs1.add(reads1, writes1);
                    }
                }

                // Finalize the current concurrent group.
                mrs0.reset();
            }

            // Expand the concurrent group with node0.
            // (Barriers were stripped, so this is always a Dispatch.)
            let _ = op_kind0; // suppress unused warning
            mrs0.add(reads0, writes0);
            result.push(i0);
        }

        // Apply the permutation to produce the reordered node list.
        let mut reordered_count = 0u32;
        for (pos, &orig_idx) in result.iter().enumerate() {
            if orig_idx != pos {
                reordered_count += 1;
            }
        }

        // Build the reordered nodes vec.
        let old_nodes = std::mem::take(&mut self.nodes);
        self.nodes = result.iter().map(|&idx| old_nodes[idx].clone()).collect();

        // Debug dump if requested.
        if std::env::var("HF2Q_REORDER_DUMP").is_ok() {
            eprintln!(
                "  [REORDER] nodes={} reordered={} ({:.1}%)",
                n,
                reordered_count,
                100.0 * reordered_count as f64 / n as f64,
            );
        }

        reordered_count
    }

    /// Get the Metal buffer pointer for a binding at the given slot index.
    ///
    /// Returns `Some(ptr)` if the slot has a `RecordedBinding::Buffer`,
    /// `None` otherwise.
    fn buffer_ptr_for_slot(bindings: &[(u64, RecordedBinding)], slot: u64) -> Option<*const std::ffi::c_void> {
        for (idx, binding) in bindings {
            if *idx == slot {
                if let RecordedBinding::Buffer { metal_buffer, offset: _ } = binding {
                    // Use the Metal buffer's GPU address as the identity key.
                    // On Apple Silicon unified memory, this uniquely identifies
                    // the allocation.
                    let ptr: *const std::ffi::c_void = metal_buffer.as_ptr() as *const _;
                    return Some(ptr);
                }
            }
        }
        None
    }

    /// Clone the binding at the given slot index.
    fn get_binding(bindings: &[(u64, RecordedBinding)], slot: u64) -> Option<RecordedBinding> {
        for (idx, binding) in bindings {
            if *idx == slot {
                return Some(binding.clone());
            }
        }
        None
    }

    /// Map a norm pipeline to its fused norm+mul pipeline name.
    ///
    /// The pipeline's `label()` is set by Metal to the function name, so we
    /// can match on it.  Returns `None` if the pipeline is not a known norm.
    fn fused_pipeline_name(pipeline: &metal::ComputePipelineState) -> Option<&'static str> {
        match pipeline.label() {
            "rms_norm_f32" => Some("rms_norm_mul_f32"),
            "rms_norm_f16" => Some("rms_norm_mul_f16"),
            "rms_norm_bf16" => Some("rms_norm_mul_bf16"),
            _ => None,
        }
    }
}

impl Default for ComputeGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Dual-buffer encoding helpers (Phase 4e.4)
// ---------------------------------------------------------------------------

/// Find the node index where the n0-th dispatch starts.
///
/// Counts `CapturedNode::Dispatch` nodes until `n0` are reached, then returns
/// the index of the n0-th dispatch (i.e., the first node of the second chunk).
/// If `n0 >= dispatch_count`, returns `nodes.len()` (everything in chunk 0).
fn find_dispatch_split_index(nodes: &[CapturedNode], n0: usize) -> usize {
    let mut dispatches_seen = 0usize;
    for (i, node) in nodes.iter().enumerate() {
        if matches!(node, CapturedNode::Dispatch { .. }) {
            dispatches_seen += 1;
            if dispatches_seen == n0 {
                return i + 1; // split AFTER the n0-th dispatch
            }
        }
    }
    nodes.len()
}

/// Encode a slice of captured nodes into a command encoder, recomputing
/// barriers on the fly from each node's read/write buffer ranges.
///
/// This is the chunked counterpart of `ComputeGraph::encode_with_barriers()`.
/// Factored out so both halves of a dual-buffer encode can use it.
///
/// Returns the number of barriers emitted.
fn encode_chunk_with_barriers(nodes: &[CapturedNode], encoder: &mut CommandEncoder) -> u32 {
    let mut tracker = ReorderConflictTracker::new();
    let mut barrier_count = 0u32;

    for node in nodes {
        match node {
            CapturedNode::Dispatch {
                pipeline,
                bindings,
                threads_per_grid,
                threads_per_threadgroup,
                threadgroup_memory,
                dispatch_kind,
                reads,
                writes,
                ..
            } => {
                let has_ranges = !reads.is_empty() || !writes.is_empty();
                if has_ranges && tracker.conflicts(reads, writes) {
                    encoder.memory_barrier();
                    tracker.reset();
                    barrier_count += 1;
                }
                if has_ranges {
                    tracker.add(reads, writes);
                }
                encoder.replay_dispatch(
                    pipeline,
                    bindings,
                    threadgroup_memory,
                    *threads_per_grid,
                    *threads_per_threadgroup,
                    *dispatch_kind,
                );
            }
            CapturedNode::Barrier => {
                encoder.memory_barrier();
                tracker.reset();
                barrier_count += 1;
            }
        }
    }
    barrier_count
}

// ---------------------------------------------------------------------------
// ReorderConflictTracker — range-based conflict detection for the reorder pass
// ---------------------------------------------------------------------------

/// Memory range conflict tracker for the reorder pass (Phase 4e.3).
///
/// Works with `MemRange` tuples `(start, end)` stored on `CapturedNode::Dispatch`,
/// rather than requiring live `&MlxBuffer` references.  This is the reorder-time
/// equivalent of the runtime `ConflictTracker`.
///
/// Conflict rules match llama.cpp's `ggml_mem_ranges_check`:
/// - Two read ranges: OK (read-read is concurrent-safe)
/// - A new read overlapping an existing write: CONFLICT (RAW)
/// - A new write overlapping any existing range: CONFLICT (WAR/WAW)
struct ReorderConflictTracker {
    /// (start, end, is_write) for all ranges in the tracked set.
    ranges: Vec<(usize, usize, bool)>,
}

impl ReorderConflictTracker {
    fn new() -> Self {
        Self {
            ranges: Vec::with_capacity(64),
        }
    }

    fn reset(&mut self) {
        self.ranges.clear();
    }

    /// Check if a dispatch with the given read/write ranges conflicts with
    /// any range in this tracker.
    fn conflicts(&self, reads: &[MemRange], writes: &[MemRange]) -> bool {
        // New reads vs existing writes (RAW)
        for &(r_start, r_end) in reads {
            for &(s, e, is_write) in &self.ranges {
                if is_write && r_start < e && r_end > s {
                    return true;
                }
            }
        }
        // New writes vs all existing ranges (WAR/WAW)
        for &(w_start, w_end) in writes {
            for &(s, e, _) in &self.ranges {
                if w_start < e && w_end > s {
                    return true;
                }
            }
        }
        false
    }

    /// Add read and write ranges to the tracked set.
    fn add(&mut self, reads: &[MemRange], writes: &[MemRange]) {
        for &(start, end) in reads {
            self.ranges.push((start, end, false));
        }
        for &(start, end) in writes {
            self.ranges.push((start, end, true));
        }
    }
}

/// Batched Metal dispatch — encodes multiple ops into a single `CommandEncoder`.
///
/// Create one per model (or per forward-pass loop).  Call [`begin`](Self::begin)
/// at the start of each forward pass to get a [`GraphSession`] that holds the
/// shared encoder.
pub struct GraphExecutor {
    device: MlxDevice,
}

impl GraphExecutor {
    /// Create a new graph executor backed by the given device.
    pub fn new(device: MlxDevice) -> Self {
        Self { device }
    }

    /// Begin a new forward pass (direct-dispatch mode).
    ///
    /// Returns a [`GraphSession`] that holds a fresh `CommandEncoder`.  All ops
    /// encoded through the session share this single encoder.  Call
    /// [`GraphSession::finish`] to commit and wait.
    pub fn begin(&self) -> Result<GraphSession<'_>> {
        let encoder = self.device.command_encoder()?;
        Ok(GraphSession {
            encoder,
            device: &self.device,
            barrier_count: 0,
            tracker: ConflictTracker::new(),
            dispatch_in_group: 0,
            total_dispatches: 0,
            group_sizes: [0; 8],
            recording: false,
        })
    }

    /// Begin a new forward pass in capture (record) mode.
    ///
    /// All op calls are recorded into a `ComputeGraph` instead of being
    /// dispatched to Metal.  When [`GraphSession::finish`] is called, the
    /// recorded graph is replayed into a fresh encoder via
    /// `ComputeGraph::encode_sequential()`.
    ///
    /// The API is identical to `begin()` — callers do not need to change
    /// any op call code.  The only behavioral difference: GPU work happens
    /// at `finish()` time rather than at each op call.
    pub fn begin_recorded(&self) -> Result<GraphSession<'_>> {
        let mut encoder = self.device.command_encoder()?;
        encoder.start_capture();
        Ok(GraphSession {
            encoder,
            device: &self.device,
            barrier_count: 0,
            tracker: ConflictTracker::new(),
            dispatch_in_group: 0,
            total_dispatches: 0,
            group_sizes: [0; 8],
            recording: true,
        })
    }

    /// Borrow the underlying device.
    pub fn device(&self) -> &MlxDevice {
        &self.device
    }
}

/// A single forward pass execution context.
///
/// All ops are encoded into one `CommandEncoder`.  Call [`finish`](Self::finish)
/// to commit the command buffer and wait for GPU completion — this is the ONLY
/// sync point per forward pass.
///
/// If an op returns an error, the session can be dropped without committing.
/// The underlying command buffer is abandoned (never committed to the GPU).
/// Tracks buffer address ranges for automatic barrier elision.
///
/// Mirrors llama.cpp's `ggml_mem_ranges` — accumulates the read and write
/// ranges of all dispatches in the current concurrent group. When a new
/// dispatch's reads overlap with an existing write (RAW), or its writes
/// overlap with an existing read or write (WAR/WAW), a barrier is needed.
/// Otherwise the dispatch can run concurrently and the barrier is elided.
///
/// Uses CPU-visible `contents_ptr()` addresses, which on Apple Silicon
/// unified memory equal the GPU addresses.
pub struct ConflictTracker {
    /// (start, end, is_write) tuples for the current concurrent group.
    ranges: Vec<(usize, usize, bool)>,
}

impl ConflictTracker {
    fn new() -> Self {
        Self {
            ranges: Vec::with_capacity(32),
        }
    }

    /// Reset the tracker — called after emitting a barrier.
    fn reset(&mut self) {
        self.ranges.clear();
    }

    /// Check if a new dispatch with the given reads and writes conflicts
    /// with the current concurrent group.
    ///
    /// Conflict rules (same as llama.cpp `ggml_mem_ranges_check`):
    /// - Two SRC (read) ranges in the same buffer: OK (read-read)
    /// - A new SRC overlapping an existing DST: CONFLICT (RAW)
    /// - A new DST overlapping an existing SRC or DST: CONFLICT (WAR/WAW)
    /// Check for conflicts and return the reason if one is found.
    /// Returns (conflict_type, new_buf_ptr, existing_buf_ptr) or None.
    fn conflicts_reason(&self, reads: &[&MlxBuffer], writes: &[&MlxBuffer])
        -> Option<(&'static str, usize, usize)>
    {
        // Check new reads against existing writes (RAW)
        for r in reads {
            let r_start = r.contents_ptr() as usize;
            let r_end = r_start + r.byte_len();
            for &(s, e, is_write) in &self.ranges {
                if is_write && r_start < e && r_end > s {
                    return Some(("RAW", r_start, s));
                }
            }
        }
        // Check new writes against existing reads and writes (WAR/WAW)
        for w in writes {
            let w_start = w.contents_ptr() as usize;
            let w_end = w_start + w.byte_len();
            for &(s, e, is_write) in &self.ranges {
                if w_start < e && w_end > s {
                    let kind = if is_write { "WAW" } else { "WAR" };
                    return Some((kind, w_start, s));
                }
            }
        }
        None
    }

    /// Add read and write ranges to the current concurrent group.
    fn add(&mut self, reads: &[&MlxBuffer], writes: &[&MlxBuffer]) {
        for r in reads {
            let start = r.contents_ptr() as usize;
            let end = start + r.byte_len();
            self.ranges.push((start, end, false));
        }
        for w in writes {
            let start = w.contents_ptr() as usize;
            let end = start + w.byte_len();
            self.ranges.push((start, end, true));
        }
    }
}

pub struct GraphSession<'a> {
    encoder: CommandEncoder,
    device: &'a MlxDevice,
    barrier_count: u32,
    tracker: ConflictTracker,
    dispatch_in_group: u32,
    total_dispatches: u32,
    /// Histogram: group_sizes[i] = number of concurrent groups with (i+1) dispatches
    group_sizes: [u32; 8],
    /// Whether this session was created in capture/record mode.
    recording: bool,
}

impl<'a> GraphSession<'a> {
    /// Encode an RMS normalization into this session's encoder.
    ///
    /// Delegates to [`ops::rms_norm::dispatch_rms_norm`].
    pub fn rms_norm(
        &mut self,
        registry: &mut KernelRegistry,
        device: &metal::DeviceRef,
        input: &MlxBuffer,
        weight: &MlxBuffer,
        output: &MlxBuffer,
        params_buf: &MlxBuffer,
        rows: u32,
        dim: u32,
    ) -> Result<()> {
        ops::rms_norm::dispatch_rms_norm(
            &mut self.encoder,
            registry,
            device,
            input,
            weight,
            output,
            params_buf,
            rows,
            dim,
        )
    }

    /// Encode a quantized matrix multiplication into this session's encoder.
    ///
    /// Delegates to [`ops::quantized_matmul::quantized_matmul`].
    /// Returns the freshly allocated output buffer.
    pub fn quantized_matmul(
        &mut self,
        registry: &mut KernelRegistry,
        device: &MlxDevice,
        input: &MlxBuffer,
        weight: &MlxBuffer,
        scales: &MlxBuffer,
        biases: &MlxBuffer,
        params: &ops::quantized_matmul::QuantizedMatmulParams,
    ) -> Result<MlxBuffer> {
        ops::quantized_matmul::quantized_matmul(
            &mut self.encoder,
            registry,
            device,
            input,
            weight,
            scales,
            biases,
            params,
        )
    }

    /// Encode a SIMD-optimized quantized matmul into this session's encoder.
    ///
    /// Delegates to [`ops::quantized_matmul::quantized_matmul_simd`].
    /// Returns the freshly allocated output buffer.
    pub fn quantized_matmul_simd(
        &mut self,
        registry: &mut KernelRegistry,
        device: &MlxDevice,
        input: &MlxBuffer,
        weight: &MlxBuffer,
        scales: &MlxBuffer,
        biases: &MlxBuffer,
        params: &ops::quantized_matmul::QuantizedMatmulParams,
    ) -> Result<MlxBuffer> {
        ops::quantized_matmul::quantized_matmul_simd(
            &mut self.encoder,
            registry,
            device,
            input,
            weight,
            scales,
            biases,
            params,
        )
    }

    /// Encode a GGML block-format quantized mat-vec into this session's encoder.
    ///
    /// Delegates to [`ops::quantized_matmul_ggml::quantized_matmul_ggml`].
    pub fn quantized_matmul_ggml(
        &mut self,
        registry: &mut KernelRegistry,
        device: &MlxDevice,
        input: &MlxBuffer,
        weight: &MlxBuffer,
        output: &mut MlxBuffer,
        params: &ops::quantized_matmul_ggml::GgmlQuantizedMatmulParams,
    ) -> Result<()> {
        ops::quantized_matmul_ggml::quantized_matmul_ggml(
            &mut self.encoder,
            registry,
            device,
            input,
            weight,
            output,
            params,
        )
    }

    /// Encode an expert-routed GGML block-format quantized mat-vec into this session's encoder.
    ///
    /// Delegates to [`ops::quantized_matmul_id_ggml::quantized_matmul_id_ggml`].
    #[allow(clippy::too_many_arguments)]
    pub fn quantized_matmul_id_ggml(
        &mut self,
        registry: &mut KernelRegistry,
        device: &MlxDevice,
        input: &MlxBuffer,
        weight: &MlxBuffer,
        ids: &MlxBuffer,
        output: &mut MlxBuffer,
        params: &ops::quantized_matmul_id_ggml::GgmlQuantizedMatmulIdParams,
    ) -> Result<()> {
        ops::quantized_matmul_id_ggml::quantized_matmul_id_ggml(
            &mut self.encoder,
            registry,
            device,
            input,
            weight,
            ids,
            output,
            params,
        )
    }

    /// Encode scaled dot-product attention into this session's encoder.
    ///
    /// Delegates to [`ops::sdpa::sdpa`].
    pub fn sdpa(
        &mut self,
        registry: &mut KernelRegistry,
        device: &MlxDevice,
        q: &MlxBuffer,
        k: &MlxBuffer,
        v: &MlxBuffer,
        output: &MlxBuffer,
        params: &ops::sdpa::SdpaParams,
        batch_size: u32,
    ) -> Result<()> {
        ops::sdpa::sdpa(
            &mut self.encoder,
            registry,
            device,
            q,
            k,
            v,
            output,
            params,
            batch_size,
        )
    }

    /// Encode flash attention vector (SIMD-vectorized decode-path SDPA).
    ///
    /// Delegates to [`ops::flash_attn_vec::flash_attn_vec`].
    pub fn flash_attn_vec(
        &mut self,
        registry: &mut KernelRegistry,
        device: &MlxDevice,
        q: &MlxBuffer,
        k: &MlxBuffer,
        v: &MlxBuffer,
        output: &MlxBuffer,
        tmp: &MlxBuffer,
        params: &ops::flash_attn_vec::FlashAttnVecParams,
    ) -> Result<()> {
        ops::flash_attn_vec::flash_attn_vec(
            &mut self.encoder,
            registry,
            device,
            q,
            k,
            v,
            output,
            tmp,
            params,
        )
    }

    /// Encode an elementwise add into this session's encoder.
    ///
    /// Delegates to [`ops::elementwise::elementwise_add`].
    pub fn elementwise_add(
        &mut self,
        registry: &mut KernelRegistry,
        device: &metal::DeviceRef,
        a: &MlxBuffer,
        b: &MlxBuffer,
        output: &MlxBuffer,
        n_elements: usize,
        dtype: DType,
    ) -> Result<()> {
        ops::elementwise::elementwise_add(
            &mut self.encoder,
            registry,
            device,
            a,
            b,
            output,
            n_elements,
            dtype,
        )
    }

    /// Encode an elementwise multiply into this session's encoder.
    ///
    /// Delegates to [`ops::elementwise::elementwise_mul`].
    pub fn elementwise_mul(
        &mut self,
        registry: &mut KernelRegistry,
        device: &metal::DeviceRef,
        a: &MlxBuffer,
        b: &MlxBuffer,
        output: &MlxBuffer,
        n_elements: usize,
        dtype: DType,
    ) -> Result<()> {
        ops::elementwise::elementwise_mul(
            &mut self.encoder,
            registry,
            device,
            a,
            b,
            output,
            n_elements,
            dtype,
        )
    }

    /// Encode a RoPE transform into this session's encoder.
    ///
    /// Delegates to [`ops::rope::dispatch_rope`].
    pub fn rope(
        &mut self,
        registry: &mut KernelRegistry,
        device: &metal::DeviceRef,
        input: &MlxBuffer,
        output: &MlxBuffer,
        params_buf: &MlxBuffer,
        positions_buf: &MlxBuffer,
        seq_len: u32,
        head_dim: u32,
    ) -> Result<()> {
        ops::rope::dispatch_rope(
            &mut self.encoder,
            registry,
            device,
            input,
            output,
            params_buf,
            positions_buf,
            seq_len,
            head_dim,
        )
    }

    /// Encode a GELU activation into this session's encoder.
    ///
    /// Delegates to [`ops::gelu::dispatch_gelu`].
    pub fn gelu(
        &mut self,
        registry: &mut KernelRegistry,
        device: &metal::DeviceRef,
        input: &MlxBuffer,
        output: &MlxBuffer,
    ) -> Result<()> {
        ops::gelu::dispatch_gelu(
            &mut self.encoder,
            registry,
            device,
            input,
            output,
        )
    }

    /// Encode a softmax into this session's encoder.
    ///
    /// Delegates to [`ops::softmax::dispatch_softmax`].
    pub fn softmax(
        &mut self,
        registry: &mut KernelRegistry,
        device: &metal::DeviceRef,
        input: &MlxBuffer,
        output: &MlxBuffer,
        params_buf: &MlxBuffer,
        rows: u32,
        cols: u32,
    ) -> Result<()> {
        ops::softmax::dispatch_softmax(
            &mut self.encoder,
            registry,
            device,
            input,
            output,
            params_buf,
            rows,
            cols,
        )
    }

    /// Encode a softcap into this session's encoder.
    ///
    /// Delegates to [`ops::softcap::dispatch_softcap`].
    pub fn softcap(
        &mut self,
        registry: &mut KernelRegistry,
        device: &metal::DeviceRef,
        input: &MlxBuffer,
        output: &MlxBuffer,
        params_buf: &MlxBuffer,
        cap: f32,
    ) -> Result<()> {
        ops::softcap::dispatch_softcap(
            &mut self.encoder,
            registry,
            device,
            input,
            output,
            params_buf,
            cap,
        )
    }

    /// Encode an RMS norm without learned scale (f32) into this session's encoder.
    ///
    /// Delegates to [`ops::rms_norm::dispatch_rms_norm_no_scale_f32`].
    pub fn rms_norm_no_scale_f32(
        &mut self,
        registry: &mut KernelRegistry,
        device: &metal::DeviceRef,
        input: &MlxBuffer,
        output: &MlxBuffer,
        params_buf: &MlxBuffer,
        rows: u32,
        dim: u32,
    ) -> Result<()> {
        ops::rms_norm::dispatch_rms_norm_no_scale_f32(
            &mut self.encoder,
            registry,
            device,
            input,
            output,
            params_buf,
            rows,
            dim,
        )
    }

    /// Encode a NeoX RoPE (f32) with optional freq_factors into this session's encoder.
    ///
    /// Delegates to [`ops::rope::dispatch_rope_neox_f32`].
    #[allow(clippy::too_many_arguments)]
    pub fn rope_neox_f32(
        &mut self,
        registry: &mut KernelRegistry,
        device: &metal::DeviceRef,
        input: &MlxBuffer,
        output: &MlxBuffer,
        params_buf: &MlxBuffer,
        positions_buf: &MlxBuffer,
        freq_factors: Option<&MlxBuffer>,
        seq_len: u32,
        n_heads: u32,
        head_dim: u32,
        rope_dim: u32,
    ) -> Result<()> {
        ops::rope::dispatch_rope_neox_f32(
            &mut self.encoder,
            registry,
            device,
            input,
            output,
            params_buf,
            positions_buf,
            freq_factors,
            seq_len,
            n_heads,
            head_dim,
            rope_dim,
        )
    }

    /// Insert a GPU memory barrier (MTLBarrierScopeBuffers).
    ///
    /// Unconditional barrier — always emits. Use `barrier_between` for
    /// automatic conflict detection that can elide unnecessary barriers.
    #[inline]
    pub fn barrier(&mut self) {
        // Record the outgoing group size
        if self.dispatch_in_group > 0 {
            let idx = (self.dispatch_in_group as usize).min(self.group_sizes.len()) - 1;
            self.group_sizes[idx] += 1;
        }
        self.encoder.memory_barrier();
        self.tracker.reset();
        self.barrier_count += 1;
        self.dispatch_in_group = 0;
    }

    /// Smart barrier with conflict detection.
    ///
    /// Checks if the next dispatch (with the given read and write buffers)
    /// actually conflicts with any dispatch in the current concurrent group.
    /// If yes, emits a Metal barrier and resets the tracker. If no, the
    /// barrier is elided and the dispatch can run concurrently.
    ///
    /// This mirrors llama.cpp's `ggml_metal_op_concurrency_check` +
    /// `ggml_metal_op_concurrency_reset` pattern.
    #[inline]
    pub fn barrier_between(&mut self, reads: &[&MlxBuffer], writes: &[&MlxBuffer]) {
        // In capture mode, stash the read/write ranges so the next captured
        // dispatch node carries them for the reorder pass (Phase 4e.3).
        if self.recording {
            let read_ranges: Vec<MemRange> = reads
                .iter()
                .map(|b| {
                    let start = b.contents_ptr() as usize;
                    (start, start + b.byte_len())
                })
                .collect();
            let write_ranges: Vec<MemRange> = writes
                .iter()
                .map(|b| {
                    let start = b.contents_ptr() as usize;
                    (start, start + b.byte_len())
                })
                .collect();
            self.encoder.set_pending_buffer_ranges(read_ranges, write_ranges);
        }

        let reason = self.tracker.conflicts_reason(reads, writes);
        if let Some((_kind, _new_ptr, _existing_ptr)) = reason {
            // Record the outgoing group size before resetting
            if self.dispatch_in_group > 0 {
                let idx = (self.dispatch_in_group as usize).min(self.group_sizes.len()) - 1;
                self.group_sizes[idx] += 1;
            }
            self.encoder.memory_barrier();
            self.tracker.reset();
            self.barrier_count += 1;
            self.dispatch_in_group = 0;
        }
        self.dispatch_in_group += 1;
        self.total_dispatches += 1;
        self.tracker.add(reads, writes);
    }

    /// Print group size histogram to stderr (for HF2Q_MLX_TIMING debug).
    pub fn dump_group_stats(&self) {
        // Record the final (unterminated) group
        let mut gs = self.group_sizes;
        if self.dispatch_in_group > 0 {
            let idx = (self.dispatch_in_group as usize).min(gs.len()) - 1;
            gs[idx] += 1;
        }
        let total_groups: u32 = gs.iter().sum();
        eprintln!("  [GROUP_STATS] dispatches={} barriers={} groups={} ratio={:.2}",
            self.total_dispatches, self.barrier_count, total_groups,
            if total_groups > 0 { self.total_dispatches as f64 / total_groups as f64 } else { 0.0 });
        for (i, &count) in gs.iter().enumerate() {
            if count > 0 {
                eprintln!("    size {}: {} groups", i + 1, count);
            }
        }
    }

    /// Register a dispatch's buffer ranges without checking for conflicts.
    ///
    /// Use after dispatching an op that doesn't need a barrier check (e.g.,
    /// the first dispatch in a session, or dispatches known to be concurrent).
    ///
    /// In recording mode, also retroactively annotates the most recently
    /// captured dispatch node with these ranges if it was missing them.
    /// That keeps the reorder pass able to reason about dispatches that
    /// were preceded by `track_dispatch` rather than `barrier_between`.
    #[inline]
    pub fn track_dispatch(&mut self, reads: &[&MlxBuffer], writes: &[&MlxBuffer]) {
        if self.recording {
            let read_ranges: Vec<MemRange> = reads
                .iter()
                .map(|b| {
                    let start = b.contents_ptr() as usize;
                    (start, start + b.byte_len())
                })
                .collect();
            let write_ranges: Vec<MemRange> = writes
                .iter()
                .map(|b| {
                    let start = b.contents_ptr() as usize;
                    (start, start + b.byte_len())
                })
                .collect();
            self.encoder
                .annotate_last_dispatch_if_missing(read_ranges, write_ranges);
        }
        self.tracker.add(reads, writes);
    }

    /// Return the number of barriers inserted so far in this session.
    #[inline]
    pub fn barrier_count(&self) -> u32 {
        self.barrier_count
    }

    /// Cumulative nanoseconds spent in ConflictTracker checks (diagnostic).
    /// Returns 0 when timing is not compiled in.
    pub fn tracker_overhead_ns(&self) -> u64 {
        0
    }

    /// Borrow the underlying command encoder for direct op dispatch.
    ///
    /// Use this when you need to call an op function that is not wrapped by
    /// a `GraphSession` method.  The returned encoder is the same shared
    /// encoder — all dispatches still go into the same command buffer.
    pub fn encoder_mut(&mut self) -> &mut CommandEncoder {
        &mut self.encoder
    }

    /// Borrow the device reference.
    pub fn device(&self) -> &MlxDevice {
        self.device
    }

    /// Whether this session is in capture/record mode.
    pub fn is_recording(&self) -> bool {
        self.recording
    }

    /// Commit the command buffer and wait for GPU completion.
    ///
    /// This is the ONLY sync point per forward pass.  After this call, all
    /// output buffers are readable by the CPU.
    ///
    /// In recording mode: extracts the captured graph, replays it into
    /// the encoder via `ComputeGraph::encode_sequential()`, then commits
    /// and waits.  The result is identical to the direct-dispatch path.
    ///
    /// Consumes the session — no further ops can be encoded.
    pub fn finish(mut self) -> Result<()> {
        if self.recording {
            if let Some(nodes) = self.encoder.take_capture() {
                let graph = ComputeGraph::from_nodes(nodes);
                graph.encode_sequential(&mut self.encoder);
            }
        }
        self.encoder.commit_and_wait()
    }

    /// Commit the command buffer WITHOUT waiting.
    ///
    /// The GPU begins executing immediately.  Use this for fire-and-forget
    /// dispatch when you do not need results until later.
    ///
    /// In recording mode: replays the captured graph before committing.
    ///
    /// Consumes the session.
    pub fn commit(mut self) -> CommandEncoder {
        if self.recording {
            if let Some(nodes) = self.encoder.take_capture() {
                let graph = ComputeGraph::from_nodes(nodes);
                graph.encode_sequential(&mut self.encoder);
            }
        }
        self.encoder.commit();
        self.encoder
    }

    /// Commit the command buffer and wait, returning split timing.
    ///
    /// Returns `(encoding_ns, gpu_wait_ns)` where:
    /// - `encoding_ns` is the time from session begin to commit (CPU encoding)
    /// - `gpu_wait_ns` is the time from commit to GPU completion
    ///
    /// The `session_begin` instant should be captured right after `exec.begin()`.
    ///
    /// In recording mode: replays the captured graph before committing.
    ///
    /// Consumes the session.
    pub fn finish_with_timing(mut self, session_begin: std::time::Instant) -> Result<(u64, u64)> {
        if self.recording {
            if let Some(nodes) = self.encoder.take_capture() {
                let graph = ComputeGraph::from_nodes(nodes);
                graph.encode_sequential(&mut self.encoder);
            }
        }
        let commit_start = std::time::Instant::now();
        let encoding_ns = commit_start.duration_since(session_begin).as_nanos() as u64;
        self.encoder.commit();
        self.encoder.wait_until_completed()?;
        let gpu_wait_ns = commit_start.elapsed().as_nanos() as u64;
        Ok((encoding_ns, gpu_wait_ns))
    }

    /// Finish with fusion: run the RMS norm + MUL fusion pass before
    /// replaying the graph.
    ///
    /// Only meaningful in recording mode.  In direct-dispatch mode, this
    /// behaves identically to `finish()`.
    ///
    /// Returns `(fusions_applied,)` on success.
    pub fn finish_with_fusion(
        mut self,
        registry: &mut KernelRegistry,
        device: &metal::DeviceRef,
    ) -> Result<u32> {
        let mut fusions = 0;
        if self.recording {
            if let Some(nodes) = self.encoder.take_capture() {
                let mut graph = ComputeGraph::from_nodes(nodes);
                fusions = graph.fuse(registry, device)?;
                graph.encode_sequential(&mut self.encoder);
            }
        }
        self.encoder.commit_and_wait()?;
        Ok(fusions)
    }

    /// Finish with fusion and split timing.
    ///
    /// Like `finish_with_timing` but runs the fusion pass first.
    /// Returns `(encoding_ns, gpu_wait_ns, fusions_applied)`.
    pub fn finish_with_fusion_and_timing(
        mut self,
        registry: &mut KernelRegistry,
        device: &metal::DeviceRef,
        session_begin: std::time::Instant,
    ) -> Result<(u64, u64, u32)> {
        let mut fusions = 0;
        if self.recording {
            if let Some(nodes) = self.encoder.take_capture() {
                let mut graph = ComputeGraph::from_nodes(nodes);
                fusions = graph.fuse(registry, device)?;
                graph.encode_sequential(&mut self.encoder);
            }
        }
        let commit_start = std::time::Instant::now();
        let encoding_ns = commit_start.duration_since(session_begin).as_nanos() as u64;
        self.encoder.commit();
        self.encoder.wait_until_completed()?;
        let gpu_wait_ns = commit_start.elapsed().as_nanos() as u64;
        Ok((encoding_ns, gpu_wait_ns, fusions))
    }

    /// Finish with fusion AND reorder: run both graph optimization passes
    /// before replaying the graph.
    ///
    /// Only meaningful in recording mode.  In direct-dispatch mode, this
    /// behaves identically to `finish()`.
    ///
    /// Returns `(fusions_applied, nodes_reordered)` on success.
    pub fn finish_with_fusion_and_reorder(
        mut self,
        registry: &mut KernelRegistry,
        device: &metal::DeviceRef,
    ) -> Result<(u32, u32)> {
        let mut fusions = 0;
        let mut reordered = 0;
        if self.recording {
            if let Some(nodes) = self.encoder.take_capture() {
                let mut graph = ComputeGraph::from_nodes(nodes);
                fusions = graph.fuse(registry, device)?;
                reordered = graph.reorder();
                graph.encode_with_barriers(&mut self.encoder);
            }
        }
        self.encoder.commit_and_wait()?;
        Ok((fusions, reordered))
    }

    /// Finish with fusion, reorder, and split timing.
    ///
    /// Like `finish_with_fusion_and_timing` but also runs the reorder pass.
    /// Returns `(encoding_ns, gpu_wait_ns, fusions_applied, nodes_reordered)`.
    pub fn finish_with_fusion_reorder_and_timing(
        mut self,
        registry: &mut KernelRegistry,
        device: &metal::DeviceRef,
        session_begin: std::time::Instant,
    ) -> Result<(u64, u64, u32, u32)> {
        let mut fusions = 0;
        let mut reordered = 0;
        if self.recording {
            if let Some(nodes) = self.encoder.take_capture() {
                let mut graph = ComputeGraph::from_nodes(nodes);
                fusions = graph.fuse(registry, device)?;
                reordered = graph.reorder();
                graph.encode_with_barriers(&mut self.encoder);
            }
        }
        let commit_start = std::time::Instant::now();
        let encoding_ns = commit_start.duration_since(session_begin).as_nanos() as u64;
        self.encoder.commit();
        self.encoder.wait_until_completed()?;
        let gpu_wait_ns = commit_start.elapsed().as_nanos() as u64;
        Ok((encoding_ns, gpu_wait_ns, fusions, reordered))
    }

    /// Finish with the full optimization pipeline: fuse, reorder, dual-buffer
    /// encode.
    ///
    /// Runs the fusion pass, reorder pass, then encodes the graph into two
    /// Metal command buffers for CPU/GPU overlap.  The first ~10% of dispatches
    /// are committed immediately so the GPU can start executing while the CPU
    /// encodes the remaining ~90%.
    ///
    /// Only meaningful in recording mode.  In direct-dispatch mode, this
    /// behaves identically to `finish()`.
    ///
    /// Returns `(fusions_applied, nodes_reordered, barriers_buf0, barriers_buf1)`.
    pub fn finish_optimized(
        mut self,
        registry: &mut KernelRegistry,
        device: &metal::DeviceRef,
    ) -> Result<(u32, u32, u32, u32)> {
        let mut fusions = 0;
        let mut reordered = 0;
        let mut barriers0 = 0u32;
        let mut barriers1 = 0u32;

        if self.recording {
            if let Some(nodes) = self.encoder.take_capture() {
                // Commit the capture encoder's empty command buffer so its
                // MTLCommandQueue pool slot is freed (same fix as timing variant).
                self.encoder.commit();

                let mut graph = ComputeGraph::from_nodes(nodes);
                fusions = graph.fuse(registry, device)?;
                reordered = graph.reorder();

                let mut enc0 = self.device.command_encoder()?;
                let mut enc1 = self.device.command_encoder()?;

                let (b0, b1) = graph.encode_dual_buffer(&mut enc0, &mut enc1);
                barriers0 = b0;
                barriers1 = b1;

                // enc0 was already committed inside encode_dual_buffer.
                // Commit enc1 and wait — Metal queue ordering guarantees enc0
                // finishes before enc1 starts executing.
                enc1.commit_and_wait()?;

                // The original encoder was never committed (capture mode drained
                // it). We need to end it cleanly — dropping it will end the
                // active encoder if any, and the uncommitted command buffer is
                // abandoned.  That is safe: Metal silently drops uncommitted
                // command buffers.
                return Ok((fusions, reordered, barriers0, barriers1));
            }
        }

        // Direct-dispatch fallback: just commit the original encoder.
        self.encoder.commit_and_wait()?;
        Ok((fusions, reordered, barriers0, barriers1))
    }

    /// Finish with the full optimization pipeline and split timing.
    ///
    /// Like `finish_optimized` but returns timing information.
    /// Returns `(encoding_ns, gpu_wait_ns, fusions, reordered, barriers_buf0, barriers_buf1)`.
    ///
    /// Timing breakdown:
    /// - `encoding_ns`: CPU time from session begin to first buffer commit
    ///   (fusion + reorder + encode chunk 0)
    /// - `gpu_wait_ns`: wall time from second buffer commit to GPU completion
    ///   (includes GPU execution of both buffers, overlapped with chunk 1 encoding)
    pub fn finish_optimized_with_timing(
        mut self,
        registry: &mut KernelRegistry,
        device: &metal::DeviceRef,
        session_begin: std::time::Instant,
    ) -> Result<(u64, u64, u32, u32, u32, u32)> {
        let mut fusions = 0;
        let mut reordered = 0;
        let mut barriers0 = 0u32;
        let mut barriers1 = 0u32;

        if self.recording {
            if let Some(nodes) = self.encoder.take_capture() {
                // Commit the capture encoder's empty command buffer so its
                // MTLCommandQueue pool slot is freed.  Without this, each
                // token leaks one uncommitted buffer and the queue exhausts
                // its ~64-slot pool after ~64 tokens, causing a deadlock.
                self.encoder.commit();

                let opt_t0 = std::time::Instant::now();
                let mut graph = ComputeGraph::from_nodes(nodes);
                let fuse_t0 = std::time::Instant::now();
                fusions = graph.fuse(registry, device)?;
                let fuse_us = fuse_t0.elapsed().as_micros();

                let reorder_t0 = std::time::Instant::now();
                let unannotated = graph.unannotated_dispatch_count();
                if unannotated == 0 {
                    reordered = graph.reorder();
                } else if std::env::var("HF2Q_MLX_TIMING").is_ok() {
                    eprintln!("  [GRAPH_OPT] WARN: skipping reorder — {} of {} dispatches lack range annotations",
                        unannotated, graph.dispatch_count());
                }
                let reorder_us = reorder_t0.elapsed().as_micros();
                let opt_us = opt_t0.elapsed().as_micros();

                let diag = std::env::var("HF2Q_GRAPH_DIAG").is_ok();
                let t0 = std::time::Instant::now();
                let mut enc0 = self.device.command_encoder()?;
                let mut enc1 = self.device.command_encoder()?;
                let enc_create_us = t0.elapsed().as_micros();

                let t1 = std::time::Instant::now();
                let (b0, b1) = graph.encode_dual_buffer(&mut enc0, &mut enc1);
                barriers0 = b0;
                barriers1 = b1;
                let encode_us = t1.elapsed().as_micros();

                let encoding_ns = session_begin.elapsed().as_nanos() as u64;

                let wait_start = std::time::Instant::now();
                enc1.commit_and_wait()?;
                let gpu_wait_ns = wait_start.elapsed().as_nanos() as u64;

                if diag {
                    eprintln!("  [DIAG] fuse={:.1}ms reorder={:.1}ms opt_total={:.1}ms enc_create={:.1}ms encode={:.1}ms gpu_wait={:.1}ms barriers={}+{}",
                        fuse_us as f64 / 1e3, reorder_us as f64 / 1e3, opt_us as f64 / 1e3,
                        enc_create_us as f64 / 1e3, encode_us as f64 / 1e3,
                        gpu_wait_ns as f64 / 1e6, b0, b1);
                }

                return Ok((encoding_ns, gpu_wait_ns, fusions, reordered, barriers0, barriers1));
            }
        }

        // Direct-dispatch fallback.
        let commit_start = std::time::Instant::now();
        let encoding_ns = commit_start.duration_since(session_begin).as_nanos() as u64;
        self.encoder.commit();
        self.encoder.wait_until_completed()?;
        let gpu_wait_ns = commit_start.elapsed().as_nanos() as u64;
        Ok((encoding_ns, gpu_wait_ns, fusions, reordered, barriers0, barriers1))
    }
}
