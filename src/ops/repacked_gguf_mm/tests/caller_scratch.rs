use super::*;
    #[test]
    fn caller_scratch_sizing_separates_immutable_and_execution_owners() {
        for ggml_type in [GgmlType::Q4_K, GgmlType::Q6_K] {
            for (n, k, _) in SPLIT_SHAPES {
                let (n, k) = (n as u32, k as u32);
                let planes = repacked_gguf_weight_plane_bytes(n, k, ggml_type).unwrap();
                let scratch = repacked_gguf_mm_scratch_bytes(
                    n, k, ggml_type, 1, RepackedGgufMmExecution::Auto,
                ).unwrap();
                assert_eq!(planes + scratch, repacked_gguf_weight_bytes(n, k, ggml_type).unwrap());
                assert_eq!(repacked_gguf_mm_scratch_bytes(
                    n, k, ggml_type, 1, RepackedGgufMmExecution::OnePass,
                ).unwrap(), 0);
                let serial_rows = (split_grid_row_tiles(n, k, ggml_type,
                    target_splits(n, k, ggml_type)) + 1) * TILE_ROWS;
                assert_eq!(repacked_gguf_mm_scratch_bytes(
                    n, k, ggml_type, serial_rows, RepackedGgufMmExecution::Auto,
                ).unwrap(), 0);
                assert!(repacked_gguf_mm_scratch_bytes(
                    n, k, ggml_type, 0, RepackedGgufMmExecution::OnePass,
                ).is_err());
            }
        }
        assert!(repacked_gguf_weight_plane_bytes(63, 256, GgmlType::Q4_K).is_err());
        assert!(repacked_gguf_mm_scratch_bytes(
            64, 256, GgmlType::Q5_K, 1, RepackedGgufMmExecution::Auto,
        ).is_err());
    }

    #[cfg(mlx_native_has_metal_tensor_artifact)]
    fn planes_for(harness: &mut Harness) -> PreparedGgufWeightPlanes {
        let params = harness.params(1);
        let mut encoder = harness.device.command_encoder().unwrap();
        let planes = prepare_repacked_gguf_weight_planes(
            &mut encoder, &mut harness.registry, &harness.device, &harness.weight, &params,
        ).unwrap();
        assert_eq!(planes.plane_bytes(), repacked_gguf_weight_plane_bytes(
            params.n, params.k, params.ggml_type,
        ).unwrap());
        encoder.commit_and_wait().unwrap();
        planes
    }

    /// Real consumers, not sizing echoes: compare both ownership paths using
    /// every split count, both activation types, both schedules and tail tiles.
    #[cfg(mlx_native_has_metal_tensor_artifact)]
    #[test]
    fn caller_scratch_matches_owned_bits_and_honors_offset_guards() {
        const GUARD: usize = 16;
        const SENTINEL: u32 = 0x7fc0_dead;
        for ggml_type in [GgmlType::Q4_K, GgmlType::Q6_K] {
            for (n, k, _) in SPLIT_SHAPES {
                let mut h = Harness::new(ggml_type, n, k);
                // Prepare from a nonzero logical source offset as well.
                let source_bytes = h.weight.data_byte_len();
                let mut source = buffer(&h.device, source_bytes + 64, DType::U8,
                    vec![source_bytes + 64]);
                source.as_mut_slice::<u8>().unwrap().fill(0xa5);
                source.as_mut_slice::<u8>().unwrap()[32..32 + source_bytes]
                    .copy_from_slice(h.weight.as_slice::<u8>().unwrap());
                let source_view = source.slice_view(32, source_bytes);
                let preparation_params = h.params(1);
                let mut encoder = h.device.command_encoder().unwrap();
                let planes = prepare_repacked_gguf_weight_planes(
                    &mut encoder, &mut h.registry, &h.device, &source_view, &preparation_params,
                ).unwrap();
                encoder.commit_and_wait().unwrap();
                assert_eq!(planes.plane_bytes(), h.prepared.planes.plane_bytes());
                let capacity = planes.execution_scratch_bytes(1, RepackedGgufMmExecution::Auto).unwrap();
                let mut scratch_parent = buffer(&h.device, capacity + 2 * GUARD * 4,
                    DType::F32, vec![capacity / 4 + 2 * GUARD]);
                scratch_parent.as_mut_slice::<u32>().unwrap().fill(SENTINEL);
                let scratch = scratch_parent.slice_view((GUARD * 4) as u64, capacity / 4);
                for dtype in [DType::F32, DType::BF16] {
                    for m in [1, 31, 32, 33, 64, 157, 720] {
                        let params = h.params(m);
                        let values = h.input(&(0..m).collect::<Vec<_>>(), dtype);
                        let input_bytes = values.data_byte_len();
                        let mut input_parent = buffer(&h.device, input_bytes + 64, dtype,
                            vec![(input_bytes + 64) / dtype.size_of()]);
                        input_parent.as_mut_slice::<u8>().unwrap().fill(0xa5);
                        input_parent.as_mut_slice::<u8>().unwrap()[32..32 + input_bytes]
                            .copy_from_slice(values.as_slice::<u8>().unwrap());
                        let input = input_parent.slice_view(32, m * k);
                        let expected = h.run_input(&values);
                        assert!(expected.iter().any(|&bits| bits != 0));
                        for execution in [RepackedGgufMmExecution::Auto, RepackedGgufMmExecution::OnePass] {
                            let mut output_parent = buffer(&h.device, (m * n + 2 * GUARD) * 4,
                                DType::F32, vec![m * n + 2 * GUARD]);
                            output_parent.as_mut_slice::<u32>().unwrap().fill(SENTINEL);
                            let output = output_parent.slice_view((GUARD * 4) as u64, m * n);
                            let required = planes.execution_scratch_bytes(m as u32, execution).unwrap();
                            let mut encoder = h.device.command_encoder().unwrap();
                            dispatch_repacked_gguf_mm_with_scratch(
                                &mut encoder, &mut h.registry, &h.device, &input, &planes, &output,
                                (required > 0).then_some(&scratch), &params, execution,
                            ).unwrap();
                            encoder.commit_and_wait().unwrap();
                            assert_eq!(output.as_logical_slice::<u32>().unwrap(), expected,
                                "{ggml_type:?} {dtype:?} {n}x{k} M={m} {execution:?}");
                            let bits = output_parent.as_slice::<u32>().unwrap();
                            assert!(bits[..GUARD].iter().chain(&bits[GUARD + m * n..])
                                .all(|&bits| bits == SENTINEL));
                        }
                        let input_bits = input_parent.as_slice::<u8>().unwrap();
                        assert!(input_bits[..32].iter().chain(&input_bits[32 + input_bytes..])
                            .all(|&bits| bits == 0xa5));
                    }
                }
                let scratch_bits = scratch_parent.as_slice::<u32>().unwrap();
                assert!(scratch_bits[..GUARD].iter().chain(&scratch_bits[GUARD + capacity / 4..])
                    .all(|&bits| bits == SENTINEL));
                let source_bits = source.as_slice::<u8>().unwrap();
                assert!(source_bits[..32].iter().chain(&source_bits[32 + source_bytes..])
                    .all(|&bits| bits == 0xa5));
            }
        }
    }

    #[cfg(mlx_native_has_metal_tensor_artifact)]
    #[test]
    fn caller_scratch_unequal_queued_weights_reuse_after_last_reader_barriers() {
        for dtype in [DType::F32, DType::BF16] {
            let mut h = Harness::new(GgmlType::Q4_K, 64, 4096);
            let planes_a = planes_for(&mut h);
            let params_b = GgmlQuantizedMatmulParams {
                m: 33, n: 128, k: 2048, ggml_type: GgmlType::Q6_K,
            };
            let mut raw_b = buffer(&h.device, 128 * 8 * 210, DType::U8, vec![128, 2048]);
            fill_weight(GgmlType::Q6_K, 83, &mut raw_b);
            let mut encoder = h.device.command_encoder().unwrap();
            let owned_b = prepare_repacked_gguf_weight(
                &mut encoder, &mut h.registry, &h.device, &raw_b, &params_b,
            ).unwrap();
            let planes_b = prepare_repacked_gguf_weight_planes(
                &mut encoder, &mut h.registry, &h.device, &raw_b, &params_b,
            ).unwrap();
            encoder.commit_and_wait().unwrap();
            let input_a = h.input(&(0..31).collect::<Vec<_>>(), dtype);
            let input_b = h.input(&(71..104).collect::<Vec<_>>(), dtype).slice_view(0, 33 * 2048);
            let expected_a = h.run_input(&input_a);
            let expected_b = buffer(&h.device, 33 * 128 * 4, DType::F32, vec![33, 128]);
            let mut encoder = h.device.command_encoder().unwrap();
            dispatch_repacked_gguf_mm_typed(
                &mut encoder, &mut h.registry, &h.device, &input_b, &owned_b, &expected_b,
                &params_b, dtype, None,
            ).unwrap();
            encoder.commit_and_wait().unwrap();
            let bytes = planes_a.execution_scratch_bytes(31, RepackedGgufMmExecution::Auto).unwrap()
                .max(planes_b.execution_scratch_bytes(33, RepackedGgufMmExecution::Auto).unwrap());
            let scratch = buffer(&h.device, bytes, DType::F32, vec![bytes / 4]);
            let outputs_a: Vec<_> = (0..4).map(|_| h.output(31)).collect();
            let outputs_b: Vec<_> = (0..4).map(|_| buffer(&h.device, 33 * 128 * 4,
                DType::F32, vec![33, 128])).collect();
            let params_a = h.params(31);
            let mut encoder = h.device.command_encoder().unwrap();
            for (output_a, output_b) in outputs_a.iter().zip(&outputs_b) {
                dispatch_repacked_gguf_mm_with_scratch(
                    &mut encoder, &mut h.registry, &h.device, &input_a, &planes_a, output_a,
                    Some(&scratch), &params_a, RepackedGgufMmExecution::Auto,
                ).unwrap();
                encoder.memory_barrier();
                dispatch_repacked_gguf_mm_with_scratch(
                    &mut encoder, &mut h.registry, &h.device, &input_b, &planes_b, output_b,
                    Some(&scratch), &params_b, RepackedGgufMmExecution::Auto,
                ).unwrap();
                encoder.memory_barrier();
            }
            encoder.commit_and_wait().unwrap();
            for (a, b) in outputs_a.iter().zip(&outputs_b) {
                assert_eq!(a.as_slice::<u32>().unwrap(), expected_a);
                assert_eq!(b.as_slice::<u32>().unwrap(), expected_b.as_slice::<u32>().unwrap());
            }
        }
    }

    /// Rejected calls must leave both outputs and reusable scratch untouched,
    /// while earlier commands remain drainable and produce their original bits.
    #[cfg(mlx_native_has_metal_tensor_artifact)]
    #[test]
    fn caller_scratch_invalid_ranges_fail_before_encoding_and_preserve_drain_ownership() {
        const SENTINEL: u32 = 0x7fc0_dead;
        let mut h = Harness::new(GgmlType::Q4_K, 64, 4096);
        let planes = planes_for(&mut h);
        let params = h.params(1);
        let input = h.input(&[7], DType::F32);
        let expected = h.run_input(&input);
        let earlier_output = h.output(1);
        let mut output = h.output(1);
        output.as_mut_slice::<u32>().unwrap().fill(SENTINEL);
        let bytes = planes.execution_scratch_bytes(1, RepackedGgufMmExecution::Auto).unwrap();
        let mut scratch = buffer(&h.device, bytes, DType::F32, vec![bytes / 4]);
        scratch.as_mut_slice::<u32>().unwrap().fill(SENTINEL);
        // A large physical allocation does not grant access outside a logical
        // subview, nor does sufficient byte length excuse a malformed shape.
        let undersized = scratch.slice_view(0, bytes / 4 - 1);
        let bad_dtype = buffer(&h.device, bytes, DType::U8, vec![bytes]);
        let bad_shape = buffer(&h.device, bytes, DType::F32, vec![1]);
        let overflowing_shape = buffer(&h.device, bytes, DType::F32, vec![usize::MAX, 2]);
        let misaligned_parent = buffer(&h.device, bytes + 16, DType::F32, vec![bytes / 4 + 4]);
        let misaligned = misaligned_parent.slice_view(1, bytes / 4);
        let output_alias = scratch.slice_view(0, 64);
        let input_alias = scratch.slice_view(0, 4096);
        let plane_alias = MlxBuffer::from_raw(planes.bytes.metal_buffer().clone(),
            DType::F32, vec![planes.bytes.byte_len() / 4]);
        let source_alias = MlxBuffer::from_raw(planes._source.metal_buffer().clone(),
            DType::F32, vec![planes._source.byte_len() / 4]);
        let short_input = input.slice_view(0, 4095);
        let short_output = output.slice_view(0, 63);
        let invalid_input_shape = buffer(&h.device, 4096 * 4, DType::F32, vec![1]);
        let invalid_output_shape = buffer(&h.device, 64 * 4, DType::F32, vec![1]);
        let wrong_input_dtype = buffer(&h.device, 4096 * 4, DType::F16, vec![8192]);
        let wrong_output_dtype = buffer(&h.device, 64 * 4, DType::U8, vec![256]);
        let misaligned_input = misaligned_parent.slice_view(1, 4096);
        let misaligned_output = misaligned_parent.slice_view(1, 64);
        let mut encoder = h.device.command_encoder().unwrap();
        h.encode(&mut encoder, &input, &earlier_output);
        let mut reject = |input: &MlxBuffer, output: &MlxBuffer,
                          scratch: Option<&MlxBuffer>, params: &GgmlQuantizedMatmulParams,
                          message: &str| {
            let result = dispatch_repacked_gguf_mm_with_scratch(
                &mut encoder, &mut h.registry, &h.device, input, &planes, output,
                scratch, params, RepackedGgufMmExecution::Auto,
            );
            assert!(matches!(&result, Err(MlxError::InvalidArgument(text))
                if text.contains(message)), "expected {message}: {result:?}");
        };
        for invalid in [None, Some(&undersized), Some(&bad_dtype), Some(&bad_shape),
            Some(&overflowing_shape), Some(&misaligned)]
        {
            reject(&input, &output, invalid, &params, "scratch");
        }
        reject(&input, &output_alias, Some(&scratch), &params, "scratch/output");
        reject(&input_alias, &output, Some(&scratch), &params, "scratch/input");
        reject(&input, &output, Some(&plane_alias), &params, "scratch/planes");
        reject(&input, &output, Some(&source_alias), &params, "scratch/source");
        reject(&input, &input.slice_view(0, 64), Some(&scratch), &params, "output/input");
        reject(&input, &plane_alias.slice_view(0, 64), Some(&scratch), &params, "output/planes");
        reject(&input, &source_alias.slice_view(0, 64), Some(&scratch), &params, "output/source");
        reject(&short_input, &output, Some(&scratch), &params, "input");
        reject(&input, &short_output, Some(&scratch), &params, "output");
        reject(&invalid_input_shape, &output, Some(&scratch), &params, "input");
        reject(&input, &invalid_output_shape, Some(&scratch), &params, "output");
        reject(&wrong_input_dtype, &output, Some(&scratch), &params, "F32/BF16");
        reject(&input, &wrong_output_dtype, Some(&scratch), &params, "F32 output");
        reject(&misaligned_input, &output, Some(&scratch), &params, "input");
        reject(&input, &misaligned_output, Some(&scratch), &params, "output");
        let wrong_params = GgmlQuantizedMatmulParams { n: 128, ..params };
        reject(&input, &output, Some(&scratch), &wrong_params, "parameters");
        let zero_rows = GgmlQuantizedMatmulParams { m: 0, ..params };
        reject(&input, &output, Some(&scratch), &zero_rows, "M >= 1");
        drop(reject);
        encoder.commit_and_wait().unwrap();
        assert_eq!(earlier_output.as_slice::<u32>().unwrap(), expected);
        assert!(output.as_slice::<u32>().unwrap().iter().all(|&bits| bits == SENTINEL));
        assert!(scratch.as_slice::<u32>().unwrap().iter().all(|&bits| bits == SENTINEL));
        // The same immutable planes remain usable after draining all errors.
        let mut encoder = h.device.command_encoder().unwrap();
        dispatch_repacked_gguf_mm_with_scratch(
            &mut encoder, &mut h.registry, &h.device, &input, &planes, &output,
            Some(&scratch), &params, RepackedGgufMmExecution::Auto,
        ).unwrap();
        encoder.commit_and_wait().unwrap();
        assert_eq!(output.as_slice::<u32>().unwrap(), expected);
    }

    #[cfg(mlx_native_has_metal_tensor_artifact)]
    #[test]
    fn caller_scratch_pipeline_errors_encode_no_partial_product_and_drain_prior_work() {
        const SENTINEL: u32 = 0x7fc0_dead;
        let mut h = Harness::new(GgmlType::Q4_K, 64, 4096);
        let planes = planes_for(&mut h);
        let params = h.params(33);
        let input = h.input(&(0..33).collect::<Vec<_>>(), DType::BF16);
        let expected = h.run_input(&input);
        let earlier_output = h.output(33);
        let mut output = h.output(33);
        let bytes = planes.execution_scratch_bytes(33, RepackedGgufMmExecution::Auto).unwrap();
        assert!(bytes > 0);
        let mut scratch = buffer(&h.device, bytes, DType::F32, vec![bytes / 4]);
        for kernel in [Q4K_MM_BF16_INPUT_KERNEL,
            "hf2q_repacked_mm_q4k_bf16_input_f32_edge", REDUCE_KERNEL]
        {
            output.as_mut_slice::<u32>().unwrap().fill(SENTINEL);
            scratch.as_mut_slice::<u32>().unwrap().fill(SENTINEL);
            let mut encoder = h.device.command_encoder().unwrap();
            h.encode(&mut encoder, &input, &earlier_output);
            h.registry.inject_next_pipeline_lookup_failure(kernel);
            let result = dispatch_repacked_gguf_mm_with_scratch(
                &mut encoder, &mut h.registry, &h.device, &input, &planes, &output,
                Some(&scratch), &params, RepackedGgufMmExecution::Auto,
            );
            assert!(matches!(&result, Err(MlxError::KernelNotFound(name)) if name == kernel));
            encoder.commit_and_wait().unwrap();
            assert_eq!(earlier_output.as_slice::<u32>().unwrap(), expected);
            assert!(output.as_slice::<u32>().unwrap().iter().all(|&bits| bits == SENTINEL));
            assert!(scratch.as_slice::<u32>().unwrap().iter().all(|&bits| bits == SENTINEL));
        }
    }

    #[cfg(mlx_native_has_metal_tensor_artifact)]
    #[test]
    fn caller_scratch_mapped_writable_arguments_are_rejected_and_prior_work_drains() {
        const VALUE: f32 = 1.25;
        let mut h = Harness::new(GgmlType::Q4_K, 64, 4096);
        let planes = planes_for(&mut h);
        let params = h.params(1);
        let input = h.input(&[13], DType::F32);
        let expected = h.run_input(&input);
        let earlier_output = h.output(1);
        let output = h.output(1);
        let bytes = planes.execution_scratch_bytes(1, RepackedGgufMmExecution::Auto).unwrap();
        let scratch = buffer(&h.device, bytes, DType::F32, vec![bytes / 4]);
        let elements = bytes / 4 + 8;
        // A real unrelated GGUF F32 mapping, with a nonzero tensor offset.
        let name = "unrelated.readonly";
        let mut file = Vec::new();
        file.extend_from_slice(b"GGUF");
        file.extend_from_slice(&3u32.to_le_bytes());
        file.extend_from_slice(&1u64.to_le_bytes());
        file.extend_from_slice(&0u64.to_le_bytes());
        file.extend_from_slice(&(name.len() as u64).to_le_bytes());
        file.extend_from_slice(name.as_bytes());
        file.extend_from_slice(&1u32.to_le_bytes());
        file.extend_from_slice(&(elements as u64).to_le_bytes());
        file.extend_from_slice(&0u32.to_le_bytes());
        file.extend_from_slice(&32u64.to_le_bytes());
        while file.len() % 32 != 0 {
            file.push(0);
        }
        file.extend_from_slice(&[0; 32]);
        for _ in 0..elements {
            file.extend_from_slice(&VALUE.to_le_bytes());
        }
        let path = std::env::temp_dir().join(format!(
            "mlx_repacked_caller_scratch_readonly_{}.gguf", std::process::id()
        ));
        std::fs::write(&path, &file).unwrap();
        let gguf = crate::GgufFile::open(&path).unwrap();
        let mapped = gguf.load_tensor_mapped(name, &h.device).unwrap();
        drop(gguf);
        std::fs::remove_file(&path).unwrap();
        assert!(mapped.is_file_backed() && !mapped.is_cpu_writable());
        assert_ne!(mapped.byte_offset(), 0);
        let mapped_scratch = mapped.slice_view(16, bytes / 4);
        let mapped_output = mapped.slice_view(16, h.n);
        let mut encoder = h.device.command_encoder().unwrap();
        h.encode(&mut encoder, &input, &earlier_output);
        let error = dispatch_repacked_gguf_mm_with_scratch(
            &mut encoder, &mut h.registry, &h.device, &input, &planes, &output,
            Some(&mapped_scratch), &params, RepackedGgufMmExecution::Auto,
        ).unwrap_err();
        assert!(error.to_string().contains("scratch requires writable non-file-backed storage"));
        for execution in [RepackedGgufMmExecution::Auto, RepackedGgufMmExecution::OnePass] {
            let error = dispatch_repacked_gguf_mm_with_scratch(
                &mut encoder, &mut h.registry, &h.device, &input, &planes, &mapped_output,
                Some(&scratch), &params, execution,
            ).unwrap_err();
            assert!(error.to_string().contains("output requires writable non-file-backed storage"));
        }
        encoder.commit_and_wait().unwrap();
        assert_eq!(earlier_output.as_slice::<u32>().unwrap(), expected);
        assert!(output.as_slice::<u32>().unwrap().iter().all(|&bits| bits == 0));
        assert!(scratch.as_slice::<u32>().unwrap().iter().all(|&bits| bits == 0));
        assert!(mapped.as_logical_slice::<u32>().unwrap().iter()
            .all(|&bits| bits == VALUE.to_bits()));
        // Zero-scratch executions may receive None and never bind this mapping.
        let mut encoder = h.device.command_encoder().unwrap();
        dispatch_repacked_gguf_mm_with_scratch(
            &mut encoder, &mut h.registry, &h.device, &input, &planes, &output,
            None, &params, RepackedGgufMmExecution::OnePass,
        ).unwrap();
        encoder.commit_and_wait().unwrap();
        assert_eq!(output.as_slice::<u32>().unwrap(), expected);
    }

    #[cfg(mlx_native_has_metal_tensor_artifact)]
    #[test]
    fn caller_scratch_real_width_schedule_boundaries_match_owned_bits() {
        // Served-width Qwen3.6 QKV, Z and KV projection geometries. Both
        // supported formats exercise their independent split/schedule policies.
        for (n, k) in [(8192, 2048), (4096, 2048), (512, 2048)] {
            for ggml_type in [GgmlType::Q4_K, GgmlType::Q6_K] {
                let mut h = Harness::new(ggml_type, n, k);
                let planes = planes_for(&mut h);
                assert!(planes.splits() > 1);
                let boundary = planes.split_grid_row_tiles as usize * TILE_ROWS as usize;
                let bytes = planes.execution_scratch_bytes(1, RepackedGgufMmExecution::Auto).unwrap();
                let scratch = buffer(&h.device, bytes, DType::F32, vec![bytes / 4]);
                for dtype in [DType::F32, DType::BF16] {
                    for m in [boundary - 1, boundary, boundary + 1, boundary + 32] {
                        let params = h.params(m);
                        let input = h.input(&(0..m).collect::<Vec<_>>(), dtype);
                        let expected = h.run_input(&input);
                        assert!(expected.iter().any(|&bits| bits != 0));
                        let auto_bytes = planes.execution_scratch_bytes(
                            m as u32, RepackedGgufMmExecution::Auto,
                        ).unwrap();
                        assert_eq!(auto_bytes > 0, m <= boundary);
                        for execution in [RepackedGgufMmExecution::Auto, RepackedGgufMmExecution::OnePass] {
                            let output = h.output(m);
                            let required = planes.execution_scratch_bytes(m as u32, execution).unwrap();
                            let mut encoder = h.device.command_encoder().unwrap();
                            dispatch_repacked_gguf_mm_with_scratch(
                                &mut encoder, &mut h.registry, &h.device, &input, &planes, &output,
                                (required > 0).then_some(&scratch), &params, execution,
                            ).unwrap();
                            encoder.commit_and_wait().unwrap();
                            assert_eq!(output.as_slice::<u32>().unwrap(), expected,
                                "{ggml_type:?} {dtype:?} {n}x{k} M={m} {execution:?}");
                        }
                    }
                }
            }
        }
    }

    #[cfg(mlx_native_has_metal_tensor_artifact)]
    #[test]
    fn caller_scratch_distinct_owners_and_cross_encoder_consumers_match_owned_bits() {
        for ggml_type in [GgmlType::Q4_K, GgmlType::Q6_K] {
            let mut h = Harness::new(ggml_type, 512, 2048);
            // The preparation producer completes before either consumer encoder
            // is created. Queue order alone cannot submit an unsubmitted producer.
            let planes = planes_for(&mut h);
            for dtype in [DType::F32, DType::BF16] {
                let input_a = h.input(&(0..31).collect::<Vec<_>>(), dtype);
                let input_b = h.input(&(73..106).collect::<Vec<_>>(), dtype);
                let expected_a = h.run_input(&input_a);
                let expected_b = h.run_input(&input_b);
                let params_a = h.params(31);
                let params_b = h.params(33);
                let bytes = planes.execution_scratch_bytes(31, RepackedGgufMmExecution::Auto).unwrap()
                    .max(planes.execution_scratch_bytes(33, RepackedGgufMmExecution::Auto).unwrap());
                assert!(bytes > 0);
                // Each in-flight dispatch has an independently owned partial
                // buffer, so the immutable weight needs no inter-product barrier.
                let scratches: Vec<_> = (0..8).map(|_| buffer(&h.device, bytes,
                    DType::F32, vec![bytes / 4])).collect();
                let outputs: Vec<_> = (0..8).map(|i| h.output(if i % 2 == 0 { 31 } else { 33 })).collect();
                let mut encoder = h.device.command_encoder().unwrap();
                for (i, (scratch, output)) in scratches.iter().zip(&outputs).enumerate() {
                    let (input, params) = if i % 2 == 0 {
                        (&input_a, &params_a)
                    } else {
                        (&input_b, &params_b)
                    };
                    dispatch_repacked_gguf_mm_with_scratch(
                        &mut encoder, &mut h.registry, &h.device, input, &planes, output,
                        Some(scratch), params, RepackedGgufMmExecution::Auto,
                    ).unwrap();
                }
                encoder.commit_and_wait().unwrap();
                for (i, output) in outputs.iter().enumerate() {
                    let expected = if i % 2 == 0 { &expected_a } else { &expected_b };
                    assert_eq!(output.as_slice::<u32>().unwrap(), expected);
                }
                let mut encoder_a = h.device.command_encoder().unwrap();
                let mut encoder_b = h.device.command_encoder().unwrap();
                dispatch_repacked_gguf_mm_with_scratch(
                    &mut encoder_a, &mut h.registry, &h.device, &input_a, &planes, &outputs[0],
                    Some(&scratches[0]), &params_a, RepackedGgufMmExecution::Auto,
                ).unwrap();
                dispatch_repacked_gguf_mm_with_scratch(
                    &mut encoder_b, &mut h.registry, &h.device, &input_b, &planes, &outputs[1],
                    Some(&scratches[1]), &params_b, RepackedGgufMmExecution::Auto,
                ).unwrap();
                // Submit both independently owned consumers before waiting.
                encoder_a.commit();
                encoder_b.commit();
                encoder_a.wait_until_completed().unwrap();
                encoder_b.wait_until_completed().unwrap();
                assert_eq!(outputs[0].as_slice::<u32>().unwrap(), expected_a);
                assert_eq!(outputs[1].as_slice::<u32>().unwrap(), expected_b);
                // Reuse scratch from encoder A only after its last reader has
                // completed; a new encoder's local barrier alone is insufficient.
                let mut encoder_c = h.device.command_encoder().unwrap();
                dispatch_repacked_gguf_mm_with_scratch(
                    &mut encoder_c, &mut h.registry, &h.device, &input_b, &planes, &outputs[1],
                    Some(&scratches[0]), &params_b, RepackedGgufMmExecution::Auto,
                ).unwrap();
                encoder_c.commit_and_wait().unwrap();
                assert_eq!(outputs[1].as_slice::<u32>().unwrap(), expected_b);
            }
        }
    }
