use super::*;
use std::cell::Cell;
fn params(q: u32, k: u32) -> DeepSeekIndexerParams {
    DeepSeekIndexerParams {
        batch: 1,
        query_len: q,
        kv_len: k,
        start_pos: 0,
        ratio: 4,
        heads: 64,
        head_dim: 128,
        top_k: 512,
        offset: 17,
    }
}
#[test]
fn query_tiles_geometry_and_partial_banks() {
    for q in [1, 255, 256, 257, 513, 2048] {
        for k in [1023, 1024, 1025] {
            let p = params(q, k);
            let plan = deepseek_indexer_query_tile_plan(&p, 520, 4).unwrap();
            assert_eq!(plan.tiles.len(), (q as usize).div_ceil(256));
            assert_eq!(plan.tiles.iter().map(|t| t.rows).sum::<usize>(), q as usize);
            assert_eq!(
                plan.topk_elements,
                deepseek_indexer_topk_scratch_elements(1, (q as usize).min(256), k as usize)
                    .unwrap()
            );
            for t in &plan.tiles {
                assert_eq!(t.origin % 256, 0);
                assert_eq!(t.start_pos, t.origin as u32);
                assert_eq!(t.q_byte_offset, t.origin * 64 * 128 * 2);
                assert_eq!(t.weight_byte_offset, t.origin * 64 * 4);
                assert_eq!(t.output_byte_offset, t.origin * 520 * 4);
                assert_eq!(t.rows, ((q as usize) - t.origin).min(256));
            }
        }
    }
}
#[test]
fn query_tiles_endpoint_workspace_is_128_mib() {
    let plan = deepseek_indexer_query_tile_plan(&params(2048, 65536), 640, 128).unwrap();
    assert_eq!(
        (plan.score_elements * 4, plan.topk_elements * 4),
        (67108864, 67108864)
    );
    assert_eq!(1073741824 - plan.workspace_bytes, 939524096);
}
#[test]
fn query_tiles_reject_geometry_overflow_and_batch() {
    let mut p = params(257, 1025);
    for batch in [0, 2] {
        p.batch = batch;
        assert!(deepseek_indexer_query_tile_plan(&p, 520, 4).is_err());
    }
    p.batch = 1;
    p.query_len = 0;
    assert!(deepseek_indexer_query_tile_plan(&p, 520, 4).is_err());
    p.query_len = 257;
    p.start_pos = u32::MAX - 256;
    assert!(deepseek_indexer_query_tile_plan(&p, 520, 4).is_err());
    p.start_pos = 0;
    assert!(deepseek_indexer_query_tile_plan(&p, 515, 4).is_err());
    assert!(deepseek_indexer_query_tile_plan(&p, usize::MAX, usize::MAX).is_err());
    assert!(plan::checked_product(&[usize::MAX, 4]).is_err());
}
#[test]
fn query_tiles_absolute_position_and_tail_layout() {
    let mut p = params(257, 1025);
    p.start_pos = 3843;
    let plan = deepseek_indexer_query_tile_plan(&p, 520, 4).unwrap();
    assert_eq!(plan.tiles[1].start_pos, 4099);
    assert_eq!(plan.tiles[1].rows, 1);
    assert_eq!(plan.topk_row_elements, 2048);
    assert_eq!((p.start_pos + 256) / 4, 1024);
    assert_eq!((plan.tiles[1].start_pos + 1) / 4, 1025);
}
#[test]
fn query_tiles_preparation_faults_emit_zero_commands() {
    let plan = deepseek_indexer_query_tile_plan(&params(513, 1025), 520, 4).unwrap();
    for fail_tile in 0..plan.tiles.len() {
        let encoded = Cell::new(0);
        let result = prepared_dispatch(
            &plan.tiles,
            |t| {
                if t.origin / 256 == fail_tile {
                    Err(invalid("injected view/extent failure"))
                } else {
                    Ok(t.origin)
                }
            },
            || Ok(()),
            |_, _| encoded.set(encoded.get() + 1),
        );
        assert!(result.is_err());
        assert_eq!(encoded.get(), 0);
    }
    for failed_pipeline in 0..4 {
        let encoded = Cell::new(0);
        let result = prepared_dispatch(
            &plan.tiles,
            |t| Ok(t.origin),
            || {
                for pipeline in 0..4 {
                    if pipeline == failed_pipeline {
                        return Err(invalid("injected pipeline/thread failure"));
                    }
                }
                Ok(())
            },
            |_, _| encoded.set(encoded.get() + 1),
        );
        assert!(result.is_err());
        assert_eq!(encoded.get(), 0);
    }
    let encoded = Cell::new(0);
    prepared_dispatch(
        &plan.tiles,
        |t| Ok(t.origin),
        || Ok(()),
        |tiles, _| encoded.set(tiles.len()),
    )
    .unwrap();
    assert_eq!(encoded.get(), 3);
}
#[test]
fn query_tiles_alias_ranges_reject_writers_not_disjoint_views() {
    let r = [
        (100, 200),
        (100, 200),
        (120, 180),
        (200, 300),
        (300, 400),
        (400, 500),
    ];
    assert!(no_alias(&r).is_ok());
    for writer in 3..6 {
        let mut bad = r;
        bad[writer] = (150, 250);
        assert!(no_alias(&bad).is_err());
    }
}

#[test]
fn query_tiles_one_element_short_buffers_reject_before_encode() {
    for (dtype, shape) in [
        (DType::BF16, vec![1, 257, 64, 128]),
        (DType::F32, vec![1, 256, 1025]),
        (DType::I32, vec![1, 256, 2048]),
        (DType::I32, vec![1, 257, 520]),
    ] {
        let bytes = plan::checked_product(&shape).unwrap() * dtype.size_of();
        let encoded = Cell::new(0);
        let tiles = deepseek_indexer_query_tile_plan(&params(257, 1025), 520, 4)
            .unwrap()
            .tiles;
        let result = prepared_dispatch(
            &tiles,
            |_| {
                validate_spec(dtype, &shape, bytes - dtype.size_of(), dtype, &shape)?;
                Ok(())
            },
            || Ok(()),
            |_, _| encoded.set(1),
        );
        assert!(result.is_err());
        assert_eq!(encoded.get(), 0);
        assert!(validate_spec(dtype, &shape, bytes, dtype, &shape).is_ok());
    }
}

#[test]
fn query_tiles_physical_ranges_include_offsets_and_reject_bounds() {
    assert_eq!(checked_range(100, 20, 10, 30).unwrap(), (120, 130));
    for args in [
        (0, 20, 10, 30),
        (100, 20, 11, 30),
        (100, usize::MAX, 1, usize::MAX),
        (usize::MAX, 1, 1, 2),
        (usize::MAX - 1, 0, 2, 2),
    ] {
        assert!(checked_range(args.0, args.1, args.2, args.3).is_err());
    }
}

#[test]
fn query_tiles_immutable_writable_arguments_reject_with_zero_encode() {
    let tiles = deepseek_indexer_query_tile_plan(&params(257, 1025), 520, 4)
        .unwrap()
        .tiles;
    for bad_argument in 0..3 {
        for (file_backed, cpu_writable) in [(true, false), (true, true), (false, false)] {
            let encoded = Cell::new(0);
            let result = (|| {
                for argument in 0..3 {
                    let flags = if argument == bad_argument {
                        (file_backed, cpu_writable)
                    } else {
                        (false, true)
                    };
                    validate_writable(flags.0, flags.1)?;
                }
                prepared_dispatch(&tiles, |t| Ok(t.origin), || Ok(()), |_, _| encoded.set(1))
            })();
            assert!(result.is_err());
            assert_eq!(encoded.get(), 0);
        }
    }
    assert!(validate_writable(false, true).is_ok());
}
