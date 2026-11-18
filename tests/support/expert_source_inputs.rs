// Independent expected compiler inputs, not values read back from the registry.
// Pipeline identity must cover shared-header bytes as well as the entry file.
pub const EXPERT_MV_SOURCE: &str = concat!(
    include_str!("../../src/shaders/prepared_f16_tiles.h"), "\n",
    include_str!("../../src/shaders/prepared_legacy_readers.h"), "\n",
    include_str!("../../src/shaders/quantized_matmul_id_ggml.metal"),
);

pub const EXPERT_MM_SOURCE: &str = concat!(
    include_str!("../../src/shaders/prepared_f16_tiles.h"), "\n",
    include_str!("../../src/shaders/quantized_matmul_id_mm.metal"),
);

pub const EXPERT_TENSOR_MM_SOURCE: &str = concat!(
    include_str!("../../src/shaders/prepared_f16_tiles.h"), "\n",
    include_str!("../../src/shaders/quantized_matmul_id_mm_tensor.metal"),
);
