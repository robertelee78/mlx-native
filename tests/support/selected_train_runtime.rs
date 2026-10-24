include!("selected_runtime_common.rs");

fn selected_train_registry() -> (MlxDevice, KernelRegistry, String) {
    selected_environment();
    let device = MlxDevice::new().expect("mandatory Metal device, no skip");
    let mut registry = KernelRegistry::new();
    flash_attn_train::register(&mut registry);
    let (source, hash) = selected_source(flash_attn_train::FLASH_ATTN_TRAIN_FWD_SHADER_SOURCE);
    for &name in flash_attn_train::all_kernel_names_for_test() {
        registry.try_register_source(name, source).unwrap();
    }
    (device, registry, hash)
}
#[test]
fn selected_train_runtime_source_required() {
    let (device, mut registry, hash) = selected_train_registry();
    for &name in flash_attn_train::all_kernel_names_for_test() {
        selected_origin(
            &mut registry,
            &device,
            name,
            &[(200, true), (201, true), (300, false), (301, false)],
            &hash,
        );
    }
    eprintln!("SELECTED_SHADER_COMPILE completed shader=train_fwd runtime_source=true");
}
#[test]
fn selected_train_d256_causal_output_and_lse() {
    let (device, mut registry, hash) = selected_train_registry();
    let (heads, kv_heads, seq, dim) = (2usize, 1usize, 17usize, 256usize);
    let qn = heads * seq * dim;
    let kn = kv_heads * seq * dim;
    let ln = heads * seq;
    let metal_bytes = 4 * (qn + kn) + 4 * ln;
    let packet = metal_bytes + 32 * qn + 32 * kn + 64 * ln + 8 * seq * heads * seq + 1024 * 1024;
    assert!(packet <= 16 * 1024 * 1024);
    let name = "flash_attn_train_fwd_bf16_d256";
    selected_origin(
        &mut registry,
        &device,
        name,
        &[(200, false), (201, false), (300, false), (301, true)],
        &hash,
    );
    let q = f32_to_bf16(&pseudo_random_f32(SEED_VAL + 190, qn));
    let k = f32_to_bf16(&pseudo_random_f32(SEED_VAL + 191, kn));
    let v = f32_to_bf16(&pseudo_random_f32(SEED_VAL + 192, kn));
    let q_buf = alloc_bf16(&device, qn, "selected Q");
    let k_buf = alloc_bf16(&device, kn, "selected K");
    let v_buf = alloc_bf16(&device, kn, "selected V");
    let mut o_buf = alloc_bf16(&device, qn, "selected O");
    let mut l_buf = alloc_f32(&device, ln, "selected L");
    fill_bf16_buf(&q_buf, &q);
    fill_bf16_buf(&k_buf, &k);
    fill_bf16_buf(&v_buf, &v);
    o_buf
        .as_mut_slice::<bf16>()
        .unwrap()
        .fill(bf16::from_f32(1024.0));
    l_buf.as_mut_slice::<f32>().unwrap().fill(8192.0);
    let mut encoder = device.command_encoder().unwrap();
    dispatch_flash_attn_train_fwd_bf16_d256(
        &mut encoder,
        &device,
        &mut registry,
        &q_buf,
        &k_buf,
        &v_buf,
        None,
        &mut o_buf,
        &mut l_buf,
        &FlashAttnTrainParams {
            batch: 1,
            n_q_heads: heads as u32,
            n_kv_heads: kv_heads as u32,
            head_dim: dim as u32,
            q_seq_len: seq as u32,
            k_seq_len: seq as u32,
            scale: 1.0 / (dim as f32).sqrt(),
            causal: true,
        },
    )
    .unwrap();
    encoder
        .commit_and_wait()
        .expect("selected training forward completed");
    let o = bf16_to_f32(o_buf.as_slice::<bf16>().unwrap());
    let l = l_buf.as_slice::<f32>().unwrap();
    assert!(o.iter().all(|x| x.is_finite() && *x != 1024.0));
    assert!(l.iter().all(|x| x.is_finite() && *x != 8192.0));
    let (expected_o, expected_l) = oracle_for_bf16(
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
    assert_close(&o, &expected_o, 5e-3, "selected training O");
    assert_close(l, &expected_l, 3e-3, "selected training L");
    selected_origin(
        &mut registry,
        &device,
        name,
        &[(200, false), (201, false), (300, false), (301, true)],
        &hash,
    );
    eprintln!("SELECTED_SHADER_PARITY completed family=train_fwd dim=256 seq=17 causal=true metal_payload_bytes={metal_bytes} scoped_packet_bytes={packet} runtime_source=true");
}
