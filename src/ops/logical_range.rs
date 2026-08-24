//! Exact logical-buffer overlap validation for fused single-dispatch APIs.

use metal::foreign_types::ForeignType;

use crate::buffer::MlxBuffer;
use crate::error::{MlxError, Result};

#[derive(Clone, Copy)]
struct LogicalRange {
    buffer_id: usize,
    start: u64,
    end: u64,
}

impl LogicalRange {
    fn new(operation: &str, buffer: &MlxBuffer) -> Result<Self> {
        let logical_bytes = u64::try_from(buffer.data_byte_len()).map_err(|_| {
            MlxError::InvalidArgument(format!("{operation}: logical buffer length exceeds u64"))
        })?;
        let end = buffer
            .byte_offset()
            .checked_add(logical_bytes)
            .ok_or_else(|| {
                MlxError::InvalidArgument(format!(
                    "{operation}: logical buffer range overflows u64"
                ))
            })?;
        Ok(Self {
            buffer_id: buffer.metal_buffer().as_ptr() as usize,
            start: buffer.byte_offset(),
            end,
        })
    }

    fn overlaps(self, other: Self) -> bool {
        self.buffer_id == other.buffer_id && self.start < other.end && other.start < self.end
    }
}

/// Reject every write/read and write/write alias before pipeline lookup.
///
/// Fused kernels execute their logical rows concurrently, so an alias that is
/// safe across two ordered dispatches may be a WAR/WAW race inside one dispatch.
pub(super) fn validate_no_write_aliases(
    operation: &str,
    reads: &[(&str, &MlxBuffer)],
    writes: &[(&str, &MlxBuffer)],
) -> Result<()> {
    for (write_index, (write_name, write)) in writes.iter().enumerate() {
        if !write.is_cpu_writable() {
            return Err(MlxError::InvalidArgument(format!(
                "{operation}: writable buffer {write_name} is read-only"
            )));
        }
        let write_range = LogicalRange::new(operation, write)?;
        for (read_name, read) in reads {
            if write_range.overlaps(LogicalRange::new(operation, read)?) {
                return Err(MlxError::InvalidArgument(format!(
                    "{operation}: writable buffer {write_name} must not overlap read buffer {read_name}"
                )));
            }
        }
        for (other_name, other) in writes.iter().skip(write_index + 1) {
            if write_range.overlaps(LogicalRange::new(operation, other)?) {
                return Err(MlxError::InvalidArgument(format!(
                    "{operation}: writable buffers {write_name} and {other_name} must not overlap"
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::validate_no_write_aliases;
    use crate::{DType, MlxDevice};

    #[test]
    fn partially_overlapping_views_are_rejected() {
        let device = MlxDevice::new().expect("device");
        let parent = device
            .alloc_buffer(64, DType::F32, vec![16])
            .expect("parent");
        let read = parent.slice_view(0, 8);
        let write = parent.slice_view(16, 8);
        let error = validate_no_write_aliases(
            "logical_range_test",
            &[("read", &read)],
            &[("write", &write)],
        )
        .expect_err("overlapping views must fail");
        assert!(error.to_string().contains("must not overlap"));
    }
}
