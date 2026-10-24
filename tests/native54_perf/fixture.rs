use mlx_native::ops::deepseek_indexer::{
    deepseek_indexer_query_tile_plan, deepseek_indexer_topk_scratch_elements, DeepSeekIndexerParams,
};
use mlx_native::{DType, MlxBuffer, MlxDevice};
pub const Q: usize = 2048;
pub const K: usize = 63488;
pub const STRIDE: usize = 640;
pub const COLUMN: usize = 128;
pub const ROOT_BYTES: usize = 1_231_029_376;
pub const HOST_BYTES: usize = 1_571_840;
pub const Q_SEED: u64 = 0x32d90dbe781c3a65;
pub const KV_SEED: u64 = 0xc1854a2b6790def3;
pub const SAMPLES: [usize; 21] = [
    0, 1, 2, 3, 4, 255, 256, 257, 511, 512, 767, 768, 1023, 1024, 1279, 1280, 1535, 1536, 1791,
    1792, 2047,
];
const PLUS: u16 = 0x3f80;
const MINUS: u16 = 0xbf80;
const GUARD: usize = 64;
pub struct Root {
    pub buffer: MlxBuffer,
    pub data: MlxBuffer,
}
impl Root {
    fn new(device: &MlxDevice, dtype: DType, shape: Vec<usize>) -> Self {
        let elements = shape.iter().product::<usize>();
        let bytes = elements * dtype.size_of();
        let mut buffer = device
            .alloc_buffer(bytes + 128, dtype, vec![(bytes + 128) / dtype.size_of()])
            .unwrap();
        buffer.as_mut_slice::<u8>().unwrap().fill(0x5a);
        let mut data = buffer.slice_view(64, elements).with_shape(shape).unwrap();
        data.as_mut_slice::<u8>().unwrap().fill(0);
        Self { buffer, data }
    }
    pub fn check_guard(&self) {
        let bytes = self.buffer.as_slice::<u8>().unwrap();
        assert!(bytes[..GUARD].iter().all(|&v| v == 0x5a));
        assert!(bytes[bytes.len() - GUARD..].iter().all(|&v| v == 0x5a));
    }
}
pub struct Packet {
    pub q: Root,
    pub kv: Root,
    pub weights: Root,
    pub scores: [Root; 2],
    pub topk: [Root; 2],
    pub outputs: [Root; 2],
    pub kv_signs: Vec<[u64; 2]>,
    pub candidates: Vec<(u32, u32)>,
    pub expected: Vec<[i32; 512]>,
}
pub fn params() -> DeepSeekIndexerParams {
    DeepSeekIndexerParams {
        batch: 1,
        query_len: Q as u32,
        kv_len: K as u32,
        start_pos: 251904,
        ratio: 4,
        heads: 64,
        head_dim: 128,
        top_k: 512,
        offset: 128,
    }
}
fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
fn initialize_signs(buffer: &mut MlxBuffer, seed: u64) {
    for (index, word) in buffer
        .as_mut_slice::<u16>()
        .unwrap()
        .chunks_exact_mut(64)
        .enumerate()
    {
        let bits = mix(seed ^ (index as u64));
        for (bit, value) in word.iter_mut().enumerate() {
            *value = if bits & (1 << bit) == 0 { PLUS } else { MINUS };
        }
    }
}
fn sign(bits: u16) -> Option<bool> {
    match bits {
        PLUS => Some(false),
        MINUS => Some(true),
        _ => None,
    }
}
fn pack128(values: &[u16]) -> [u64; 2] {
    let mut result = [0u64; 2];
    for (feature, &value) in values.iter().enumerate() {
        if sign(value).expect("invalid actual BF16 sign") {
            result[feature / 64] |= 1 << (feature % 64);
        }
    }
    result
}
fn numerator(query: &[[u64; 2]; 64], key: [u64; 2]) -> u32 {
    query
        .iter()
        .map(|head| {
            let different = (head[0] ^ key[0]).count_ones() + (head[1] ^ key[1]).count_ones();
            (128 - 2 * different as i32).max(0) as u32
        })
        .sum()
}
fn exact_match(expected: &[i32], actual: &[i32]) -> bool {
    expected == actual
}
impl Packet {
    pub fn new(device: &MlxDevice) -> Self {
        // Frozen requested envelopes checked BEFORE any payload allocation.
        assert!(ROOT_BYTES <= 2 * 1024 * 1024 * 1024);
        assert!(HOST_BYTES <= 2 * 1024 * 1024);
        assert_eq!(std::mem::size_of::<(u32, u32)>(), 8);
        assert_eq!(K * 16 + K * 8 + 1024 + 21 * 512 * 4 + 4096, HOST_BYTES);
        let plan = deepseek_indexer_query_tile_plan(&params(), STRIDE, COLUMN).unwrap();
        let full = deepseek_indexer_topk_scratch_elements(1, Q, K).unwrap();
        let formula = Q * 64 * 128 * 2
            + K * 128 * 2
            + Q * 64 * 4
            + Q * K * 4
            + full * 4
            + plan.workspace_bytes
            + 2 * Q * STRIDE * 4
            + 9 * 128;
        assert_eq!(formula, ROOT_BYTES);
        let mut q = Root::new(device, DType::BF16, vec![1, Q, 64, 128]);
        let mut kv = Root::new(device, DType::BF16, vec![1, K, 128]);
        let mut weights = Root::new(device, DType::F32, vec![1, Q, 64]);
        initialize_signs(&mut q.data, Q_SEED);
        initialize_signs(&mut kv.data, KV_SEED);
        weights.data.as_mut_slice::<f32>().unwrap().fill(1.0 / 64.0);
        let scores = [
            Root::new(device, DType::F32, vec![1, Q, K]),
            Root::new(device, DType::F32, vec![1, 256, K]),
        ];
        let topk = [
            Root::new(device, DType::I32, vec![1, Q, full / Q]),
            Root::new(device, DType::I32, vec![1, 256, plan.topk_row_elements]),
        ];
        let outputs = [
            Root::new(device, DType::I32, vec![1, Q, STRIDE]),
            Root::new(device, DType::I32, vec![1, Q, STRIDE]),
        ];
        let mut result = Self {
            q,
            kv,
            weights,
            scores,
            topk,
            outputs,
            kv_signs: vec![[0; 2]; K],
            candidates: vec![(0, 0); K],
            expected: vec![[0; 512]; 21],
        };
        assert_eq!(
            result
                .roots()
                .map(|r| r.buffer.data_byte_len())
                .sum::<usize>(),
            ROOT_BYTES
        );
        assert_eq!(result.kv_signs.capacity(), K);
        assert_eq!(result.candidates.capacity(), K);
        assert_eq!(result.expected.capacity(), 21);
        result.prepare_oracle();
        result
    }
    pub fn roots(&self) -> impl Iterator<Item = &Root> {
        [
            &self.q,
            &self.kv,
            &self.weights,
            &self.scores[0],
            &self.scores[1],
            &self.topk[0],
            &self.topk[1],
            &self.outputs[0],
            &self.outputs[1],
        ]
        .into_iter()
    }
    fn prepare_oracle(&mut self) {
        let q = self.q.data.as_slice::<u16>().unwrap();
        assert!(q.iter().all(|&v| sign(v).is_some()));
        assert!(self
            .weights
            .data
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .all(|&v| v == 1.0 / 64.0));
        let kv = self.kv.data.as_slice::<u16>().unwrap();
        for (key, values) in kv.chunks_exact(128).enumerate() {
            self.kv_signs[key] = pack128(values);
        }
        for (sample, &row) in SAMPLES.iter().enumerate() {
            let mut query = [[0u64; 2]; 64];
            for (head, words) in query.iter_mut().enumerate() {
                *words = pack128(&q[(row * 64 + head) * 128..(row * 64 + head + 1) * 128]);
            }
            let valid = K.min((251904 + row + 1) / 4);
            for key in 0..valid {
                self.candidates[key] = (numerator(&query, self.kv_signs[key]), key as u32);
            }
            self.candidates[..valid].sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
            for rank in 0..512 {
                self.expected[sample][rank] = self.candidates[rank].1 as i32 + 128;
            }
        }
        // CPU-only falsification controls: never mutate real input/output.
        assert!(sign(0).is_none());
        let mut changed = self.expected[0];
        changed[0] ^= 1;
        assert!(!exact_match(&self.expected[0], &changed));
    }
    pub fn poison(&mut self, arm: usize) {
        self.outputs[arm]
            .data
            .as_mut_slice::<i32>()
            .unwrap()
            .fill(-7);
    }
    pub fn check_arm(&self, arm: usize) {
        let output = self.outputs[arm].data.as_slice::<i32>().unwrap();
        for row in 0..Q {
            let values = &output[row * STRIDE..(row + 1) * STRIDE];
            assert!(values[..COLUMN].iter().all(|&v| v == -7));
            let valid = K.min((251904 + row + 1) / 4) as i32;
            assert!(values[COLUMN..]
                .iter()
                .all(|&v| v >= 128 && v < 128 + valid));
        }
        for (sample, &row) in SAMPLES.iter().enumerate() {
            assert!(exact_match(
                &self.expected[sample],
                &output[row * STRIDE + COLUMN..(row + 1) * STRIDE]
            ));
        }
        for root in self.roots() {
            root.check_guard();
        }
    }
    pub fn check_pair(&self) {
        assert_eq!(
            self.outputs[0].data.as_slice::<i32>().unwrap(),
            self.outputs[1].data.as_slice::<i32>().unwrap()
        );
    }
}
