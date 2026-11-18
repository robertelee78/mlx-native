use super::*;

const PACKET: &str = include_str!("../fixtures/prepared-legacy-mv-raw-output-goldens.json");
const PACKET_SHA: &str = "81de5eb614af0b9131faca38b65bc0f2ca1fbf5679f643912cbd88470aff7ebc";
const MAX_WORDS: usize = 4 * 8 * 512;

struct GoldenCheck<'a> {
    device: &'a MlxDevice,
    records: &'a [Value],
    next: usize,
    raw: Guarded,
    canonical: Guarded,
    scratch: IdMmScratch,
}

impl<'a> GoldenCheck<'a> {
    fn new(device: &'a MlxDevice, records: &'a [Value]) -> Self {
        Self { device, records, next: 0,
            raw: Guarded::new(device, MAX_WORDS, DType::F32),
            canonical: Guarded::new(device, MAX_WORDS, DType::F32),
            scratch: IdMmScratch::alloc_with_top_k(device, 64, 4, 8).unwrap() }
    }

    fn compare(&mut self, registry: &mut KernelRegistry, source: &Source,
        source_hash: &str, planes: &CompletedGroupedM8Experts, input: &Guarded,
        ids: &Guarded, p: &GgmlQuantizedMatmulIdParams, routing: &GgmlRoutingPolicy,
        kind: Option<MvKind>, masked: bool) {
        let record = &self.records[self.next];
        let label = match kind { None => "Auto", Some(MvKind::Stock) => "Stock",
            Some(MvKind::Nr2) => "Nr2", Some(MvKind::Geglu) => "Geglu" };
        let cols = if matches!(kind, Some(MvKind::Geglu)) { p.n as usize / 2 } else { p.n as usize };
        let rows = p.n_tokens as usize * p.top_k as usize;
        let words = rows * cols;
        let input_words = p.n_tokens as usize * p.k as usize;
        assert_eq!(record["kind"], label, "record {}", self.next);
        assert_eq!(record["ggml_type"], format!("{:?}", p.ggml_type));
        for (key, value) in [("n_experts", u64::from(p.n_experts)), ("n", u64::from(p.n)),
            ("k", u64::from(p.k)), ("tokens", u64::from(p.n_tokens)),
            ("top_k", u64::from(p.top_k)), ("expert_stride", p.expert_stride),
            ("output_cols", cols as u64), ("output_words", words as u64)] {
            assert_eq!(record[key].as_u64(), Some(value), "record {} {key}", self.next);
        }
        assert_eq!(record["masked"], masked);
        assert_eq!(record["q6_nr2"], routing.expert_q6k_mv_nr2);
        assert_eq!(record["case"]["source"]["raw_sha256"], source_hash);
        let input_hash = input.digest(input_words);
        let ids_hash = ids.digest(rows);
        assert_eq!(record["input_sha256"], input_hash);
        assert_eq!(record["ids_sha256"], ids_hash);
        self.raw.reset(words, 0x7fc5_6666);
        self.canonical.reset(words, 0x7fc5_6666);
        let iv = input.view(input_words);
        let idv = ids.view(rows);
        let rv = self.raw.view(words);
        let pv = self.canonical.view(words);
        let mut encoder = self.device.command_encoder().unwrap();
        if let Some(kind) = kind {
            mv_dispatch(false, kind, masked, &mut encoder, registry, self.device,
                &iv, source, planes, &idv, &rv, p, routing);
            mv_dispatch(true, kind, masked, &mut encoder, registry, self.device,
                &iv, source, planes, &idv, &pv, p, routing);
        } else {
            quantized_matmul_id_ggml_pooled_with_policy(&mut encoder, registry,
                self.device, &iv, &source.view, &idv, &rv, &mut self.scratch, p, routing).unwrap();
            quantized_matmul_id_ggml_prepared_with_policy(&mut encoder, registry,
                self.device, &iv, planes, &idv, &pv, &mut self.scratch, p, routing).unwrap();
        }
        encoder.commit_and_wait().unwrap();
        let expected = record["output_sha256"].as_str().unwrap();
        assert_eq!(self.raw.digest(words), expected,
            "original raw golden record={} kind={label} type={:?}", self.next, p.ggml_type);
        assert_eq!(self.canonical.digest(words), expected,
            "canonical original golden record={} kind={label} type={:?}", self.next, p.ggml_type);
        self.raw.check(words); self.canonical.check(words);
        input.check(input_words); ids.check(rows);
        assert_eq!(input.digest(input_words), input_hash, "activation mutation");
        assert_eq!(ids.digest(rows), ids_hash, "ID mutation");
        self.next += 1;
    }
}

fn frozen_policy(nr2: bool) -> GgmlRoutingPolicy {
    let mut routing = policy(false, false, nr2);
    routing.expert_mm_threshold = 32;
    routing
}

fn fill_dense(input: &mut Guarded, rng: &mut Rng) {
    let words = input.capacity;
    for value in input.payload_mut(words) {
        *value = (((rng.next() >> 40) as f32 / 16_777_216.0) * 3.75 - 1.875).to_bits();
    }
}

fn raw_hash(source: &Source) -> String {
    format!("{:x}", Sha256::digest(source.view.as_slice::<u8>().unwrap()))
}

fn auto_first(check: &mut GoldenCheck<'_>, ids: &mut Guarded, ty: GgmlType) {
    let device = check.device;
    let base = params(ty, 512, 2048, 64, 256, 1);
    let mut rng = Rng(0xa17e_1d66_0994_2318 ^ 1);
    let first = Source::new(device, &base, &mut rng, false);
    let second = Source::new(device, &base, &mut rng, false);
    let source_hash = raw_hash(&first);
    let p = GgmlQuantizedMatmulIdParams { n_tokens: 1, ..base };
    let mut input = Guarded::new(device, 2048, DType::F32);
    let mut slotted = Guarded::new(device, 2048, DType::F32);
    fill_dense(&mut input, &mut rng);
    fill_dense(&mut slotted, &mut rng);
    for kind in [None, Some(MvKind::Stock)] {
        if ty == GgmlType::Q4_K && kind.is_some() { continue; }
        let routing = frozen_policy(kind.is_none());
        let mut registry = KernelRegistry::new();
        registry.freeze_ggml_routing_policy(routing).unwrap();
        let planes = prepare(device, &mut registry, &first, &base);
        for invalid in [None, Some(64u32), Some(u32::MAX)] {
            ids.reset(1, 0);
            if let Some(id) = invalid { ids.payload_mut(1)[0] = id; }
            check.compare(&mut registry, &first, &source_hash, &planes, &input,
                ids, &p, &routing, kind, false);
        }
    }
    first.unchanged(); second.unchanged();
    input.check(2048); slotted.check(2048);
}

fn dense_corpus(check: &mut GoldenCheck<'_>, ids: &mut Guarded, ty: GgmlType) {
    let device = check.device;
    for (k, tokens) in [(256usize, 1usize), (768, 4)] {
        let p = params(ty, 512, k, 64, tokens, 8);
        let mut rng = Rng(0x619a_56ce_33a2_8891 ^ k as u64 ^ 512u64);
        let source = Source::new(device, &p, &mut rng, true);
        let source_hash = raw_hash(&source);
        let rows = tokens * 8;
        let inputs = std::array::from_fn::<_, 3, _>(|activation| {
            let mut input = Guarded::new(device, tokens * k, DType::F32);
            fill_dense(&mut input, &mut rng);
            if activation != 0 {
                input.payload_mut(tokens * k)[(tokens - 1) * k + 7] =
                    if activation == 1 { f32::NAN.to_bits() } else { f32::INFINITY.to_bits() };
            }
            input
        });
        for kind in [MvKind::Stock, MvKind::Nr2, MvKind::Geglu] {
            if ty != GgmlType::Q4_K && matches!(kind, MvKind::Geglu) { continue; }
            let routing = frozen_policy(matches!(kind, MvKind::Nr2));
            let mut registry = KernelRegistry::new();
            registry.freeze_ggml_routing_policy(routing).unwrap();
            let planes = prepare(device, &mut registry, &source, &p);
            for input in &inputs {
                for exceptional in [false, true] {
                    ids.reset(rows, 0);
                    for (pair, id) in ids.payload_mut(rows).iter_mut().enumerate() {
                        *id = if exceptional && pair % 7 == 0 {
                            if pair % 2 == 0 { 64 } else { u32::MAX }
                        } else if pair % 11 < 7 { 3 } else { (pair * 13 % 64) as u32 };
                    }
                    for masked in [false, true] {
                        if matches!(kind, MvKind::Geglu) && masked { continue; }
                        check.compare(&mut registry, &source, &source_hash, &planes,
                            input, ids, &p, &routing, Some(kind), masked);
                    }
                }
            }
        }
        source.unchanged();
    }
}

#[test]
#[ignore = "Apple M5; all117 independently frozen original raw MV cases"]
fn prepared_legacy_mv_independent_goldens() {
    assert_eq!(format!("{:x}", Sha256::digest(PACKET.as_bytes())), PACKET_SHA,
        "immutable original MV packet changed");
    let packet: Value = serde_json::from_str(PACKET).unwrap();
    let device = MlxDevice::new().unwrap();
    assert!(device.name().starts_with("Apple M5"), "original golden qualification device required");
    let mut check = GoldenCheck::new(&device, packet["records"].as_array().unwrap());
    let mut ids = Guarded::new(&device, 32, DType::U32);
    for ty in [GgmlType::Q4_K, GgmlType::Q6_K] { auto_first(&mut check, &mut ids, ty); }
    for ty in [GgmlType::Q4_K, GgmlType::Q6_K] { dense_corpus(&mut check, &mut ids, ty); }
    assert_eq!(check.next, check.records.len(), "original MV cases omitted");
}
