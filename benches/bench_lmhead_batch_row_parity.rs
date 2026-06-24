//! ADR-040 Phase F M2.2 / S1 — lm_head batch-row PARITY proof.
//!
//! S1 (batched lm_head across N decode slots) rests on ONE unproven kernel
//! property: for a fixed quantized weight, does the `mv` matmul produce output
//! ROW r BIT-IDENTICALLY whether it is computed (a) as a standalone `m=1` call
//! on input row r, or (b) as part of a fused `m=N` call? If the kernel reorders
//! its K-accumulation or tiles differently as a function of `m`, the per-row
//! logits change and batched lm_head can NEVER be bit-identical to N sequential
//! scalar decodes — which would make S1 (and the whole sliced batched-forward
//! plan) unsound. This test proves or REFUTES that property empirically BEFORE
//! any production decode code is touched (mantra: testable hypothesis first).
//!
//! Hypothesis H-S1-rowparity: `quantized_matmul_ggml` output row r at m=N is
//! bitwise-equal (compared as raw u32 bits) to the same row computed at m=1, for
//! the gemma4 lm_head shape (Q6_K, k=2816) at N in {2,4,8}.
//! Falsifier: any row differs in any bit at any N.
//!
//! Run: cargo bench -p mlx-native --bench bench_lmhead_batch_row_parity

use mlx_native::ops::quantized_matmul_ggml::{
    quantized_matmul_ggml, GgmlQuantizedMatmulParams, GgmlType,
};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

// gemma4 hidden = 2816. Use a modest output width n (the per-row determinism
// property is independent of n — the same `kernel_mul_mv_q6_K_f32` runs for any
// n — so a small n keeps the test fast while exercising the identical code path
// as the real 262144-wide lm_head).
const K: u32 = 2816;
const N: u32 = 4096;
const QTYPE: GgmlType = GgmlType::Q6_K;
const M_VALUES: &[u32] = &[2, 4, 8];

fn alloc_weight(device: &MlxDevice, n: u32, k: u32, qt: GgmlType) -> MlxBuffer {
    let blocks_per_row = (k as u64) / (qt.block_values() as u64);
    let total = (n as u64) * blocks_per_row * (qt.block_bytes() as u64);
    device
        .alloc_buffer(total as usize, DType::U8, vec![total as usize])
        .expect("alloc weight")
}
fn alloc_f32(device: &MlxDevice, n: usize) -> MlxBuffer {
    device.alloc_buffer(n * 4, DType::F32, vec![n]).expect("alloc f32")
}

/// Deterministic small finite bytes for the quantized weight. Kept small so the
/// Q6_K f16 super-block scale `d` never decodes to NaN/Inf (which would make a
/// bitwise comparison ill-defined). Values are FIXED across all runs/m.
fn fill_weight(buf: &mut MlxBuffer) {
    let dst: &mut [u8] = buf.as_mut_slice().expect("weight bytes");
    for (i, b) in dst.iter_mut().enumerate() {
        // low-amplitude, deterministic, non-degenerate
        *b = ((i * 37 + 11) % 17) as u8;
    }
}

/// Deterministic input activation value for (row, col).
fn input_val(row: usize, col: usize) -> f32 {
    // distinct per row AND per col so a row-leak would change the bits
    let x = ((row * 31 + col * 7 + 3) % 97) as f32;
    (x / 97.0) - 0.5
}

fn run_once(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    m: u32,
) {
    let params = GgmlQuantizedMatmulParams { m, n: N, k: K, ggml_type: QTYPE };
    let mut enc = device.command_encoder().expect("encoder");
    quantized_matmul_ggml(&mut enc, registry, device, input, weight, output, &params)
        .expect("qmatmul");
    enc.commit_and_wait().expect("commit");
}

fn main() {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();

    println!("S1 lm_head batch-row parity — Q6_K, k={K}, n={N}, M5 Max");
    println!("{}", "-".repeat(56));

    let mut weight = alloc_weight(&device, N, K, QTYPE);
    fill_weight(&mut weight);

    let max_m = *M_VALUES.iter().max().unwrap() as usize;

    // Reference: each row computed as a STANDALONE m=1 call.
    let mut reference: Vec<Vec<u32>> = Vec::with_capacity(max_m);
    for row in 0..max_m {
        let mut in1 = alloc_f32(&device, K as usize);
        {
            let s: &mut [f32] = in1.as_mut_slice().expect("in1");
            for c in 0..K as usize {
                s[c] = input_val(row, c);
            }
        }
        let out1 = alloc_f32(&device, N as usize);
        run_once(&device, &mut registry, &in1, &weight, &out1, 1);
        let bits: &[u32] = out1.as_slice().expect("out1 bits");
        reference.push(bits.to_vec());
    }

    let mut all_pass = true;
    for &m in M_VALUES {
        let mu = m as usize;
        // Batched input [m, k] with the SAME per-row values as the reference.
        let mut inb = alloc_f32(&device, mu * K as usize);
        {
            let s: &mut [f32] = inb.as_mut_slice().expect("inb");
            for row in 0..mu {
                for c in 0..K as usize {
                    s[row * K as usize + c] = input_val(row, c);
                }
            }
        }
        let outb = alloc_f32(&device, mu * N as usize);
        run_once(&device, &mut registry, &inb, &weight, &outb, m);
        let bits: &[u32] = outb.as_slice().expect("outb bits");

        let mut mismatch_rows = 0usize;
        let mut first_mismatch: Option<(usize, usize, u32, u32)> = None;
        for row in 0..mu {
            let base = row * N as usize;
            for col in 0..N as usize {
                let got = bits[base + col];
                let want = reference[row][col];
                if got != want {
                    mismatch_rows += 1;
                    if first_mismatch.is_none() {
                        first_mismatch = Some((row, col, want, got));
                    }
                    break;
                }
            }
        }
        let ok = mismatch_rows == 0;
        all_pass &= ok;
        println!(
            "m={m}: {} ({}/{} rows bit-identical to m=1){}",
            if ok { "PASS" } else { "FAIL" },
            mu - mismatch_rows,
            mu,
            match first_mismatch {
                Some((r, c, w, g)) =>
                    format!("  first diff row{r} col{c}: m1=0x{w:08x} mN=0x{g:08x}"),
                None => String::new(),
            }
        );
    }

    println!("{}", "-".repeat(56));
    if all_pass {
        println!(
            "H-S1-rowparity HOLDS: mv output rows are bit-identical across m.\n\
             => batched lm_head CAN be bit-identical to N sequential decodes. S1 sound."
        );
    } else {
        println!(
            "H-S1-rowparity REFUTED: mv reorders/retiles by m.\n\
             => batched lm_head can NOT match scalar bit-for-bit; S1 plan needs rework\n\
                (e.g. fixed-tile kernel, or accept logit-equivalence not bit-equivalence)."
        );
        std::process::exit(1);
    }
}
