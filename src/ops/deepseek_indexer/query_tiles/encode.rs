use super::*;

fn capture(encoder: &mut CommandEncoder, reads: &[(usize, usize)], writes: &[(usize, usize)]) {
    if encoder.is_capturing() {
        encoder.set_pending_buffer_ranges(reads.to_vec(), writes.to_vec());
    }
}

// No validation, view construction, pipeline lookup or Result after this boundary.
pub(super) fn encode(
    encoder: &mut CommandEncoder,
    pipelines: &Pipelines,
    t: &PreparedTile,
    layout: &DeepSeekIndexerOutputLayout,
) {
    let p = &t.params;
    let queries = p.query_len as usize;
    let keys = p.kv_len as usize;
    let blocks = keys.div_ceil(TOPK_BLOCK_THREADS);
    let plan = DeepSeekIndexerTopKPlan {
        block_threads: 1024,
        block_count: blocks as u32,
        list_count: blocks as u32,
        scratch_row_stride: (blocks * 512) as u32,
    };
    encoder.set_op_kind(CapturedOpKind::Other);
    capture(encoder, &t.ranges[0..3], &t.ranges[3..4]);
    encoder.encode_threadgroups_with_args(
        &pipelines.score,
        &[
            (0, KernelArg::Bytes(as_bytes(p))),
            (1, KernelArg::Buffer(&t.q)),
            (2, KernelArg::Buffer(&t.kv)),
            (3, KernelArg::Buffer(&t.weights)),
            (4, KernelArg::Buffer(&t.scratch)),
        ],
        MTLSize::new(
            keys.div_ceil(SCORE_KEYS_PER_GROUP) as u64,
            queries.div_ceil(SCORE_QUERIES_PER_GROUP) as u64,
            1,
        ),
        MTLSize::new(SCORE_THREADS, 1, 1),
    );
    encoder.memory_barrier();
    capture(encoder, &t.ranges[3..4], &t.ranges[4..5]);
    encoder.encode_threadgroups_with_args_and_shared(
        &pipelines.block,
        &[
            (0, KernelArg::Bytes(as_bytes(p))),
            (1, KernelArg::Bytes(as_bytes(&plan))),
            (2, KernelArg::Buffer(&t.scratch)),
            (3, KernelArg::Buffer(&t.bank_a)),
        ],
        &[(0, 4096)],
        MTLSize::new((queries * blocks) as u64, 1, 1),
        MTLSize::new(1024, 1, 1),
    );
    encoder.memory_barrier();
    let mut count = blocks;
    let mut input = &t.bank_a;
    let mut output = &t.bank_b;
    let mut in_range = t.ranges[4];
    let mut out_range = t.ranges[5];
    while count > 1 {
        let merge_plan = DeepSeekIndexerTopKPlan {
            list_count: count as u32,
            ..plan
        };
        let merged = count.div_ceil(2);
        capture(encoder, &[t.ranges[3], in_range], &[out_range]);
        encoder.encode_threadgroups_with_args(
            &pipelines.merge,
            &[
                (0, KernelArg::Bytes(as_bytes(p))),
                (1, KernelArg::Bytes(as_bytes(&merge_plan))),
                (2, KernelArg::Buffer(&t.scratch)),
                (3, KernelArg::Buffer(input)),
                (4, KernelArg::Buffer(output)),
            ],
            MTLSize::new((queries * merged) as u64, 1, 1),
            MTLSize::new(TOPK_MERGE_THREADS, 1, 1),
        );
        encoder.memory_barrier();
        std::mem::swap(&mut input, &mut output);
        std::mem::swap(&mut in_range, &mut out_range);
        count = merged;
    }
    let final_plan = DeepSeekIndexerTopKPlan {
        list_count: 1,
        ..plan
    };
    capture(encoder, &[in_range], &t.ranges[6..7]);
    encoder.encode_with_args(
        &pipelines.finalize,
        &[
            (0, KernelArg::Bytes(as_bytes(p))),
            (1, KernelArg::Bytes(as_bytes(&final_plan))),
            (2, KernelArg::Bytes(as_bytes(layout))),
            (3, KernelArg::Buffer(input)),
            (4, KernelArg::Buffer(&t.output)),
        ],
        MTLSize::new(512, queries as u64, 1),
        MTLSize::new(256, 1, 1),
    );
    // RAW/WAR/WAW: the next tile reuses both score and ping-pong banks.
    // Capture ranges above also expose reuse to reordered dispatch scheduling.
    encoder.memory_barrier();
}
