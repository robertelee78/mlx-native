include!("selected_runtime_common.rs");

fn selected_prefill_registry() -> (MlxDevice, KernelRegistry, String) {
    selected_environment();
    let device = MlxDevice::new().expect("mandatory Metal device, no skip");
    let mut registry = KernelRegistry::new();
    flash_attn_prefill::register(&mut registry);
    let (source, hash) = selected_source(flash_attn_prefill::FLASH_ATTN_PREFILL_SHADER_SOURCE);
    registry
        .try_register_source("flash_attn_prefill_bf16_d256", source)
        .unwrap();
    (device, registry, hash)
}
#[test]
fn selected_prefill_runtime_source_required() {
    let (device, mut registry, hash) = selected_prefill_registry();
    for flags in [
        [
            (200, true),
            (201, true),
            (300, false),
            (301, false),
            (303, false),
        ],
        [
            (200, false),
            (201, false),
            (300, false),
            (301, true),
            (303, false),
        ],
    ] {
        selected_origin(
            &mut registry,
            &device,
            "flash_attn_prefill_bf16_d256",
            &flags,
            &hash,
        );
    }
    eprintln!("SELECTED_SHADER_COMPILE completed shader=prefill runtime_source=true");
}
fn selected_qwen_causal(heads: usize, kv_heads: usize, family: &str) {
    let (device, mut registry, hash) = selected_prefill_registry();
    let (seq, dim) = (33usize, 256usize);
    let qn = heads * seq * dim;
    let kn = kv_heads * seq * dim;
    let metal_bytes = 4 * (qn + kn); // Q/K/V/O all BF16
                                     // Conservative explicit host tensor+all per-query score/exp allocations,
                                     // plus <=1MiB for source/comment/hash and test-owned scalar metadata.
    let packet = metal_bytes + 32 * qn + 32 * kn + 8 * seq * heads * seq + 1024 * 1024;
    assert!(packet <= 16 * 1024 * 1024);
    selected_origin(
        &mut registry,
        &device,
        "flash_attn_prefill_bf16_d256",
        &[
            (200, false),
            (201, false),
            (300, false),
            (301, true),
            (303, false),
        ],
        &hash,
    );
    // Existing seed/input conversion and existing CPU oracle; no copied shader math.
    let q = f32_to_bf16(&pseudo_random_f32(SEED + 190, qn));
    let k = f32_to_bf16(&pseudo_random_f32(SEED + 191, kn));
    let v = f32_to_bf16(&pseudo_random_f32(SEED + 192, kn));
    let q_buf = alloc_bf16(&device, qn, "selected Q");
    let k_buf = alloc_bf16(&device, kn, "selected K");
    let v_buf = alloc_bf16(&device, kn, "selected V");
    let mut out = alloc_bf16(&device, qn, "selected O");
    fill_bf16_buffer(&q_buf, &q);
    fill_bf16_buffer(&k_buf, &k);
    fill_bf16_buffer(&v_buf, &v);
    out.as_mut_slice::<bf16>()
        .unwrap()
        .fill(bf16::from_f32(1024.0));
    let mut encoder = device.command_encoder().unwrap();
    dispatch_flash_attn_prefill_bf16_d256(
        &mut encoder,
        &device,
        &mut registry,
        &q_buf,
        &k_buf,
        &v_buf,
        None,
        &mut out,
        &FlashAttnPrefillParams {
            n_heads: heads as u32,
            n_kv_heads: kv_heads as u32,
            head_dim: dim as u32,
            seq_len_q: seq as u32,
            seq_len_k: seq as u32,
            batch: 1,
            scale: 1.0 / (dim as f32).sqrt(),
            do_causal: true,
        },
    )
    .expect("selected causal dispatch");
    // All fallible encoding is before submission; read only after checked wait.
    encoder
        .commit_and_wait()
        .expect("selected causal command completed");
    let actual = bf16_to_f32(out.as_slice::<bf16>().unwrap());
    assert!(actual.iter().all(|v| v.is_finite() && *v != 1024.0));
    let expected = reference_for_bf16(
        &q,
        &k,
        &v,
        None,
        1,
        heads,
        kv_heads,
        seq,
        seq,
        dim,
        1.0 / (dim as f32).sqrt(),
        true,
    );
    assert_close_gpu(&actual, &expected, BF16_GPU_ATOL, BF16_GPU_RTOL, family);
    selected_origin(
        &mut registry,
        &device,
        "flash_attn_prefill_bf16_d256",
        &[
            (200, false),
            (201, false),
            (300, false),
            (301, true),
            (303, false),
        ],
        &hash,
    );
    eprintln!("SELECTED_SHADER_PARITY completed family={family} heads={heads} kv_heads={kv_heads} dim=256 seq=33 causal=true metal_payload_bytes={metal_bytes} scoped_packet_bytes={packet} runtime_source=true model_loaded=false");
}
#[test]
fn selected_qwen38_h24_kv4_d256_causal() {
    selected_qwen_causal(24, 4, "qwen38");
}
#[test]
fn selected_qwen36_h16_kv2_d256_causal() {
    selected_qwen_causal(16, 2, "qwen36");
}
