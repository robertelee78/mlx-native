use mlx_native::{DType, MlxDevice};

fn u8_buffer(device: &MlxDevice, values: &[u8]) -> mlx_native::MlxBuffer {
    let mut buffer = device
        .alloc_buffer(values.len(), DType::U8, vec![values.len()])
        .expect("allocate buffer");
    buffer
        .as_mut_slice::<u8>()
        .expect("writable u8 buffer")
        .copy_from_slice(values);
    buffer
}

#[test]
fn raw_blit_honors_explicit_and_slice_offsets_without_touching_canaries() {
    let device = MlxDevice::new().expect("Metal device");
    let source_values: Vec<u8> = (0..96).map(|i| (i * 17 + 3) as u8).collect();
    let source = u8_buffer(&device, &source_values);
    let destination = u8_buffer(&device, &[0xa5; 112]);
    let source_view = source.slice_view(7, 61);
    let destination_view = destination.slice_view(19, 73);

    let mut encoder = device.command_encoder().expect("command encoder");
    encoder
        .blit_copy_bytes(&source_view, 11, &destination_view, 13, 29)
        .expect("encode raw byte blit");
    encoder.commit_and_wait().expect("run raw byte blit");

    let result = destination.as_slice::<u8>().expect("destination bytes");
    let physical_destination = 19 + 13;
    let physical_source = 7 + 11;
    assert!(result[..physical_destination].iter().all(|&v| v == 0xa5));
    assert_eq!(
        &result[physical_destination..physical_destination + 29],
        &source_values[physical_source..physical_source + 29]
    );
    assert!(result[physical_destination + 29..]
        .iter()
        .all(|&v| v == 0xa5));
}

#[test]
fn raw_blit_supports_disjoint_same_buffer_ranges_and_rejects_overlap() {
    let device = MlxDevice::new().expect("Metal device");
    let initial: Vec<u8> = (0..96).map(|i| (255 - i) as u8).collect();
    let buffer = u8_buffer(&device, &initial);

    let mut encoder = device.command_encoder().expect("command encoder");
    encoder
        .blit_copy_bytes(&buffer, 5, &buffer, 61, 19)
        .expect("disjoint in-allocation copy");
    encoder.commit_and_wait().expect("run same-buffer blit");
    let result = buffer.as_slice::<u8>().expect("buffer bytes");
    assert_eq!(&result[61..80], &initial[5..24]);
    assert_eq!(&result[..61], &initial[..61]);
    assert_eq!(&result[80..], &initial[80..]);

    let mut rejected = device.command_encoder().expect("command encoder");
    let error = rejected
        .blit_copy_bytes(&buffer, 8, &buffer, 16, 24)
        .expect_err("overlap must fail closed");
    assert!(error.to_string().contains("overlapping same-buffer ranges"));

    // Exact self-copy is intentionally a no-op and remains legal even while
    // capture is active because no graph node needs to be represented.
    rejected.start_capture();
    rejected
        .blit_copy_bytes(&buffer, 9, &buffer, 9, 17)
        .expect("exact self-copy no-op");
    let other = u8_buffer(&device, &[0; 96]);
    let capture_error = rejected
        .blit_copy_bytes(&buffer, 0, &other, 0, 1)
        .expect_err("nonempty capture blit must fail closed");
    assert!(capture_error.to_string().contains("graph capture"));
}

#[test]
fn raw_blit_rejects_logical_view_overflow_before_encoding() {
    let device = MlxDevice::new().expect("Metal device");
    let source = u8_buffer(&device, &[7; 32]).slice_view(8, 8);
    let destination = u8_buffer(&device, &[0; 32]).slice_view(16, 8);
    let mut encoder = device.command_encoder().expect("command encoder");

    assert!(encoder
        .blit_copy_bytes(&source, 7, &destination, 0, 2)
        .is_err());
    assert!(encoder
        .blit_copy_bytes(&source, 0, &destination, 8, 1)
        .is_err());
}
