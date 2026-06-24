//! ADR-040 Phase F M2.2 / S2 — MoE `_id` matmul per-token PARITY proof.
//!
//! S2 (batched MoE, `n_tokens=N`) — and the whole `[N,hidden]` batched decode
//! body — rests on one unproven kernel property: for a fixed stacked expert
//! weight, does `quantized_matmul_id_ggml` produce token t's output block
//! BIT-IDENTICALLY whether t is computed (a) alone as an `n_tokens=1` call with
//! t's own `ids`, or (b) as part of a fused `n_tokens=N` call? If the kernel's
//! per-token K-accumulation or expert dispatch varies with `n_tokens`, the
//! batched MoE diverges from N sequential per-slot MoEs and the batched body can
//! never be bit-identical to serial — making S2/S3 unsound. Prove or REFUTE it
//! BEFORE the 2476-line encode_one_layer restructure (mantra: hypothesis first).
//!
//! Hypothesis H-S2-tokenparity: `quantized_matmul_id_ggml` output block for
//! token t at `n_tokens=N` is bitwise-equal (raw u32) to the same token computed
//! at `n_tokens=1`, for the gemma4 gate_up shape (Q6_K, k=2816, n=1408,
//! top_k=8, 128 experts) at N in {2,4,8}. Falsifier: any block differs in any
//! bit at any N.
//!
//! The dense projections (QKV/O, m=N) are already covered by H-S1-rowparity
//! (same `quantized_matmul_ggml` kernel); this proof closes the MoE leg.
//!
//! Run: cargo bench -p mlx-native --bench bench_moe_id_token_parity

use mlx_native::ops::quantized_matmul_id_ggml::{
    quantized_matmul_id_ggml, GgmlQuantizedMatmulIdParams,
};
use mlx_native::ops::quantized_matmul_ggml::GgmlType;
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

// gemma4 26B-A4B gate_up MoE shape (the n_tokens=N leg of the batched body).
const K: u32 = 2816;
const N: u32 = 1408;
const TOP_K: u32 = 8;
const N_EXPERTS: u32 = 128;
const QTYPE: GgmlType = GgmlType::Q6_K;
const N_VALUES: &[u32] = &[2, 4, 8];

fn alloc_weight_stack(device: &MlxDevice) -> (MlxBuffer, u64) {
    let blocks_per_row = (K as u64) / (QTYPE.block_values() as u64);
    let per_expert_bytes = (N as u64) * blocks_per_row * (QTYPE.block_bytes() as u64);
    let total = per_expert_bytes * (N_EXPERTS as u64);
    let mut buf = device
        .alloc_buffer(total as usize, DType::U8, vec![total as usize])
        .expect("alloc weight stack");
    // Deterministic, low-amplitude bytes (keep the Q6_K f16 super-block scale
    // finite so a bitwise compare is well-defined). FIXED across all calls.
    let dst: &mut [u8] = buf.as_mut_slice().expect("weight bytes");
    for (i, b) in dst.iter_mut().enumerate() {
        *b = ((i * 37 + 11) % 17) as u8;
    }
    (buf, per_expert_bytes)
}

fn alloc_f32(device: &MlxDevice, n: usize) -> MlxBuffer {
    device.alloc_buffer(n * 4, DType::F32, vec![n]).expect("alloc f32")
}

fn input_val(token: usize, col: usize) -> f32 {
    let x = ((token * 31 + col * 7 + 3) % 97) as f32;
    (x / 97.0) - 0.5
}

// Deterministic per-token expert ids (spread across experts, distinct per token
// so a cross-token leak would change the output bits).
fn token_ids(token: usize) -> Vec<u32> {
    (0..TOP_K as usize)
        .map(|j| (((token * 13 + j * 31 + 5) * 7919) % N_EXPERTS as usize) as u32)
        .collect()
}

fn run(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    ids: &MlxBuffer,
    output: &mut MlxBuffer,
    n_tokens: u32,
    expert_stride: u64,
) {
    let params = GgmlQuantizedMatmulIdParams {
        n_tokens,
        top_k: TOP_K,
        n: N,
        k: K,
        n_experts: N_EXPERTS,
        expert_stride,
        ggml_type: QTYPE,
    };
    let mut enc = device.command_encoder().expect("encoder");
    quantized_matmul_id_ggml(&mut enc, registry, device, input, weight, ids, output, &params)
        .expect("id matmul");
    enc.commit_and_wait().expect("commit");
}

fn main() {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();
    let (weight, expert_stride) = alloc_weight_stack(&device);

    println!("S2 MoE _id per-token parity — gemma4 gate_up Q6_K, k={K} n={N} top_k={TOP_K}, M5 Max");
    println!("{}", "-".repeat(60));

    let max_n = *N_VALUES.iter().max().unwrap() as usize;
    let block = (TOP_K as usize) * (N as usize); // output elements per token

    // Reference: each token computed ALONE as an n_tokens=1 call.
    let mut reference: Vec<Vec<u32>> = Vec::with_capacity(max_n);
    for t in 0..max_n {
        let mut in1 = alloc_f32(&device, K as usize);
        {
            let s: &mut [f32] = in1.as_mut_slice().expect("in1");
            for c in 0..K as usize {
                s[c] = input_val(t, c);
            }
        }
        let mut ids1 = device
            .alloc_buffer(TOP_K as usize * 4, DType::U32, vec![TOP_K as usize])
            .expect("ids1");
        {
            let s: &mut [u32] = ids1.as_mut_slice().expect("ids1 slice");
            s.copy_from_slice(&token_ids(t));
        }
        let mut out1 = alloc_f32(&device, block);
        run(&device, &mut registry, &in1, &weight, &ids1, &mut out1, 1, expert_stride);
        reference.push(out1.as_slice::<u32>().expect("out1 bits").to_vec());
    }

    let mut all_pass = true;
    for &n in N_VALUES {
        let nu = n as usize;
        let mut inb = alloc_f32(&device, nu * K as usize);
        {
            let s: &mut [f32] = inb.as_mut_slice().expect("inb");
            for t in 0..nu {
                for c in 0..K as usize {
                    s[t * K as usize + c] = input_val(t, c);
                }
            }
        }
        let mut idsb = device
            .alloc_buffer(nu * TOP_K as usize * 4, DType::U32, vec![nu * TOP_K as usize])
            .expect("idsb");
        {
            let s: &mut [u32] = idsb.as_mut_slice().expect("idsb slice");
            for t in 0..nu {
                s[t * TOP_K as usize..(t + 1) * TOP_K as usize].copy_from_slice(&token_ids(t));
            }
        }
        let mut outb = alloc_f32(&device, nu * block);
        run(&device, &mut registry, &inb, &weight, &idsb, &mut outb, n, expert_stride);
        let bits: &[u32] = outb.as_slice().expect("outb bits");

        let mut mismatch = 0usize;
        let mut first: Option<(usize, usize, u32, u32)> = None;
        for t in 0..nu {
            for j in 0..block {
                let got = bits[t * block + j];
                let want = reference[t][j];
                if got != want {
                    mismatch += 1;
                    if first.is_none() {
                        first = Some((t, j, want, got));
                    }
                    break;
                }
            }
        }
        let ok = mismatch == 0;
        all_pass &= ok;
        println!(
            "n_tokens={n}: {} ({}/{} token blocks bit-identical to n_tokens=1){}",
            if ok { "PASS" } else { "FAIL" },
            nu - mismatch,
            nu,
            match first {
                Some((t, j, w, g)) =>
                    format!("  first diff tok{t} elem{j}: n1=0x{w:08x} nN=0x{g:08x}"),
                None => String::new(),
            }
        );
    }

    println!("{}", "-".repeat(60));
    if all_pass {
        println!(
            "H-S2-tokenparity HOLDS: MoE _id token blocks are bit-identical across n_tokens.\n\
             => batched MoE (n_tokens=N) CAN be bit-identical to N sequential per-slot MoEs.\n\
             S2/S3 batched-body MoE leg is sound (dense leg already covered by H-S1-rowparity)."
        );
    } else {
        println!(
            "H-S2-tokenparity REFUTED: MoE _id varies by n_tokens.\n\
             => batched MoE can NOT match serial bit-for-bit; S2 needs a per-token\n\
                fixed-dispatch kernel or accept logit-equivalence not bit-equivalence."
        );
        std::process::exit(1);
    }
}
