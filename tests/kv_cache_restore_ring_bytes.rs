//! `dispatch_kv_cache_restore_ring_bytes` must invert
//! `dispatch_kv_cache_linearize_ring_bytes` bitwise on exactly the restored
//! rows (full and partial tails via `src_row_offset`) and leave every other
//! byte of the ring view and of its parent buffer unchanged.
//!
//! The test runs on the default library path and re-runs itself in a child
//! process with `MLX_PRECOMPILED_METALLIB=0` (runtime source compile).

#![cfg(target_vendor = "apple")]
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use mlx_native::ops::kv_cache_copy::{
    dispatch_kv_cache_linearize_ring_bytes, dispatch_kv_cache_restore_ring_bytes,
};
use mlx_native::{DType, KernelPipelineOrigin, KernelRegistry, MlxBuffer, MlxDevice};

const RING_CAPACITY: u32 = 1024;
const STAGING_CAPACITY: u32 = 4;
const PREFIX: usize = 37;
const SUFFIX: usize = 53;
const CANARY: u8 = 0xA5;
const CLOBBER: u8 = 0x5A;

fn next_u64(state: &mut u64) -> u64 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    state.wrapping_mul(0x2545_f491_4f6c_dd1d)
}

fn runtime_path() -> bool {
    matches!(
        std::env::var("MLX_PRECOMPILED_METALLIB").as_deref(),
        Ok("0") | Ok("false") | Ok("off")
    )
}

fn parent_with_view(device: &MlxDevice, payload: usize) -> (MlxBuffer, MlxBuffer) {
    let total = PREFIX + payload + SUFFIX;
    let mut parent = device
        .alloc_buffer(total, DType::U8, vec![total])
        .expect("parent");
    parent.as_mut_slice::<u8>().unwrap().fill(CANARY);
    let view = parent.slice_view(PREFIX as u64, payload);
    (parent, view)
}

fn ring_index(head: u32, slot: u32, row_bytes: u32) -> usize {
    PREFIX + ((head * RING_CAPACITY + slot) * row_bytes) as usize
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    ring_parent: &mut MlxBuffer,
    ring: &MlxBuffer,
    staging_parent: &mut MlxBuffer,
    staging: &MlxBuffer,
    n_heads: u32,
    row_bytes: u32,
    n_saved: u32,
    src_row_offset: u32,
    logical_end: u32,
    state: &mut u64,
) {
    // Fresh ring contents (canaries outside the view stay put).
    {
        let bytes = ring_parent.as_mut_slice::<u8>().unwrap();
        let payload = (n_heads * RING_CAPACITY * row_bytes) as usize;
        for byte in &mut bytes[PREFIX..PREFIX + payload] {
            *byte = next_u64(state) as u8;
        }
    }
    let original: Vec<u8> = ring_parent.as_slice::<u8>().unwrap().to_vec();

    // Save the pre-image of the newest `n_saved` rows.
    let mut encoder = device.command_encoder().expect("encoder");
    dispatch_kv_cache_linearize_ring_bytes(
        &mut encoder,
        registry,
        device.metal_device(),
        ring,
        staging,
        n_heads,
        RING_CAPACITY,
        STAGING_CAPACITY,
        n_saved,
        logical_end,
        row_bytes,
    )
    .expect("linearize");
    encoder.commit_and_wait().expect("commit linearize");

    // Overwrite the saved rows, as a verify forward would.
    {
        let bytes = ring_parent.as_mut_slice::<u8>().unwrap();
        for head in 0..n_heads {
            for i in 0..n_saved {
                let slot = (logical_end - n_saved + i) % RING_CAPACITY;
                let start = ring_index(head, slot, row_bytes);
                bytes[start..start + row_bytes as usize].fill(CLOBBER);
            }
        }
    }

    // Restore the tail `[src_row_offset, n_saved)` of the pre-image.
    let n_restore = n_saved - src_row_offset;
    let mut encoder = device.command_encoder().expect("encoder");
    dispatch_kv_cache_restore_ring_bytes(
        &mut encoder,
        registry,
        device.metal_device(),
        staging,
        ring,
        n_heads,
        STAGING_CAPACITY,
        RING_CAPACITY,
        src_row_offset,
        n_restore,
        logical_end,
        row_bytes,
    )
    .expect("restore");
    encoder.commit_and_wait().expect("commit restore");

    let mut expected = original;
    for head in 0..n_heads {
        for i in 0..src_row_offset {
            let slot = (logical_end - n_saved + i) % RING_CAPACITY;
            let start = ring_index(head, slot, row_bytes);
            expected[start..start + row_bytes as usize].fill(CLOBBER);
        }
    }
    let actual = ring_parent.as_slice::<u8>().unwrap();
    if actual != expected.as_slice() {
        let first = actual
            .iter()
            .zip(&expected)
            .position(|(a, e)| a != e)
            .unwrap();
        panic!(
            "heads={n_heads} row_bytes={row_bytes} n_saved={n_saved} offset={src_row_offset} \
             logical_end={logical_end}: first differing parent byte {first} \
             (actual {:#04x}, expected {:#04x})",
            actual[first], expected[first]
        );
    }
    let staging_bytes = staging_parent.as_slice::<u8>().unwrap();
    assert!(staging_bytes[..PREFIX].iter().all(|&b| b == CANARY));
    assert!(staging_bytes[staging_bytes.len() - SUFFIX..]
        .iter()
        .all(|&b| b == CANARY));
}

fn round_trip_all_cases() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let mut state = 0x4e31_5245_5354_4f52u64;
    let mut cases = 0usize;
    for row_bytes in [4u32, 256, 512, 1024] {
        for n_heads in [2u32, 8] {
            let (mut ring_parent, ring) =
                parent_with_view(&device, (n_heads * RING_CAPACITY * row_bytes) as usize);
            let (mut staging_parent, staging) =
                parent_with_view(&device, (n_heads * STAGING_CAPACITY * row_bytes) as usize);
            // No wrap (slots 596..599) and wrap (slots 1022..1023, 0..1).
            for logical_end in [600u32, 1026, 2050] {
                for n_saved in 1..=STAGING_CAPACITY {
                    for src_row_offset in 0..n_saved {
                        run_case(
                            &device,
                            &mut registry,
                            &mut ring_parent,
                            &ring,
                            &mut staging_parent,
                            &staging,
                            n_heads,
                            row_bytes,
                            n_saved,
                            src_row_offset,
                            logical_end,
                            &mut state,
                        );
                        cases += 1;
                    }
                }
            }
        }
    }
    assert_eq!(cases, 4 * 2 * 3 * 10);

    let origin = registry
        .pipeline_identity("kv_cache_restore_ring_bytes")
        .expect("identity")
        .origin;
    let want = if runtime_path() {
        KernelPipelineOrigin::RuntimeSource
    } else {
        KernelPipelineOrigin::PrecompiledMetallib
    };
    println!("runtime_path={} restore origin {origin:?}", runtime_path());
    assert_eq!(origin, want);
}

fn refuses_writes_past_view() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let row_bytes = 256u32;
    let n_heads = 8u32;
    let (_ring_parent, ring) =
        parent_with_view(&device, (n_heads * RING_CAPACITY * row_bytes) as usize);
    let (_staging_parent, staging) =
        parent_with_view(&device, (n_heads * STAGING_CAPACITY * row_bytes) as usize);
    // A view one byte short of the ring, inside a larger parent buffer: the
    // Metal buffer is big enough, the view is not.
    let short_ring = ring.slice_view(0, (n_heads * RING_CAPACITY * row_bytes) as usize - 1);
    let short_staging =
        staging.slice_view(0, (n_heads * STAGING_CAPACITY * row_bytes) as usize - 1);
    let restore = |registry: &mut KernelRegistry,
                   src: &MlxBuffer,
                   dst: &MlxBuffer,
                   offset: u32,
                   n: u32,
                   end: u32| {
        let mut encoder = device.command_encoder().expect("encoder");
        dispatch_kv_cache_restore_ring_bytes(
            &mut encoder,
            registry,
            device.metal_device(),
            src,
            dst,
            n_heads,
            STAGING_CAPACITY,
            RING_CAPACITY,
            offset,
            n,
            end,
            row_bytes,
        )
    };
    assert!(restore(&mut registry, &staging, &short_ring, 0, 1, 1026).is_err());
    assert!(restore(&mut registry, &short_staging, &ring, 0, 1, 1026).is_err());
    // Offset + rows past the staging capacity.
    assert!(restore(&mut registry, &staging, &ring, 3, 2, 1026).is_err());
    // More rows than the ring holds is impossible with a 4-row staging
    // buffer; logical_end below history_len is refused.
    assert!(restore(&mut registry, &staging, &ring, 0, 2, 1).is_err());
    assert!(restore(&mut registry, &staging, &ring, 1, 3, 1026).is_ok());
}

#[test]
fn kv_cache_restore_ring_bytes_inverts_linearize() {
    round_trip_all_cases();
    refuses_writes_past_view();

    if std::env::var_os("MLX_NATIVE_N1_RUNTIME_SOURCE_CHILD").is_none() {
        let output = std::process::Command::new(std::env::current_exe().expect("test exe"))
            .arg("--exact")
            .arg("kv_cache_restore_ring_bytes_inverts_linearize")
            .arg("--nocapture")
            .env("MLX_NATIVE_N1_RUNTIME_SOURCE_CHILD", "1")
            .env("MLX_PRECOMPILED_METALLIB", "0")
            .output()
            .expect("run runtime-source N1 gate");
        assert!(
            output.status.success(),
            "runtime-source N1 gate failed: {}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("running 1 test") && stdout.contains("runtime_path=true"),
            "runtime-source N1 gate did not run on the runtime path: {stdout}"
        );
    }
}
