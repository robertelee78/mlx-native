//! Observed backing metadata, not physical ownership, residency or release proof.
use super::MlxBuffer;
use crate::{MlxError, Result, RootOriginSnapshot};

/// Constructor provenance. Unknown includes imports and uncertified legacy roots.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum BufferProvenance {
    Owned,
    NoCopy,
    Unknown,
}

/// Actual retained readonly mmap and registered Metal CPU range. Addresses are
/// process-local observations, never dereferenceable capabilities or file IDs.
/// The mapping's slice extent excludes any rounded final-page padding, whereas
/// registered_bytes includes the actual page-rounded NoCopy resource extent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NoCopyBackingRange {
    pub mapping_base_address: usize,
    pub mapping_bytes: usize,
    pub registered_base_address: usize,
    pub registered_bytes: usize,
}

/// Immutable point-in-time descriptor. Clones/subviews share origin, not a new
/// root. `allocated_size` is Metal's reported metric, NOT unique physical RAM.
/// This value retains no Metal/mmap owner. Addresses must not be used after the
/// source resource dies; address reuse does not establish identity. A consumer
/// joining file identity must retain the buffer and its own verified file owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferBackingSnapshot {
    pub origin: RootOriginSnapshot,
    pub provenance: BufferProvenance,
    pub metal_root_bytes: usize,
    pub metal_allocated_bytes: u64,
    pub storage_mode: metal::MTLStorageMode,
    pub view_byte_offset: u64,
    pub logical_bytes: usize,
    pub no_copy: Option<NoCopyBackingRange>,
}

fn checked_span(base: usize, offset: u64, bytes: usize, extent: usize) -> Result<usize> {
    let error =
        || MlxError::InvalidArgument("backing descriptor range overflow or out of bounds".into());
    let offset = usize::try_from(offset).map_err(|_| error())?;
    let end = offset.checked_add(bytes).ok_or_else(error)?;
    if end > extent {
        return Err(error());
    }
    base.checked_add(end).ok_or_else(error)?;
    base.checked_add(offset).ok_or_else(error)
}

impl MlxBuffer {
    /// Read actual resource/view metadata without touching payload or changing
    /// authority health, allocation, residency, command ordering or raw-escape
    /// state. Existing authority refusal rules are unchanged. Unknown/Legacy
    /// origin is not promoted by this informational getter. Heap/imported roots
    /// have no certified backing descriptor and remain Unknown.
    pub fn backing_snapshot(&self) -> Result<BufferBackingSnapshot> {
        let inner = &self.storage.inner;
        let metal_root_bytes = self.byte_len();
        checked_span(0, self.byte_offset, self.data_byte_len, metal_root_bytes)?;
        let origin = self.root_origin_snapshot();
        let no_copy = self
            .storage
            .file_backing
            .as_ref()
            .map(|mapping| -> Result<NoCopyBackingRange> {
                let mapping_base_address = mapping.as_ptr() as usize;
                let registered_base_address = inner.contents() as usize;
                // Validate the logical bytes against the mmap SLICE, not padded tail.
                let relative = registered_base_address
                    .checked_sub(mapping_base_address)
                    .ok_or_else(|| {
                        MlxError::InvalidArgument("NoCopy base precedes mapping".into())
                    })?;
                let logical_offset = u64::try_from(relative)
                    .ok()
                    .and_then(|n| n.checked_add(self.byte_offset))
                    .ok_or_else(|| {
                        MlxError::InvalidArgument("NoCopy logical offset overflow".into())
                    })?;
                checked_span(
                    mapping_base_address,
                    logical_offset,
                    self.data_byte_len,
                    mapping.len(),
                )?;
                registered_base_address
                    .checked_add(metal_root_bytes)
                    .ok_or_else(|| {
                        MlxError::InvalidArgument("NoCopy registered extent overflow".into())
                    })?;
                Ok(NoCopyBackingRange {
                    mapping_base_address,
                    mapping_bytes: mapping.len(),
                    registered_base_address,
                    registered_bytes: metal_root_bytes,
                })
            })
            .transpose()?;
        let provenance = if no_copy.is_some() {
            BufferProvenance::NoCopy
        } else if matches!(origin, RootOriginSnapshot::KnownNative { .. }) {
            BufferProvenance::Owned
        } else {
            BufferProvenance::Unknown
        };
        Ok(BufferBackingSnapshot {
            origin,
            provenance,
            metal_root_bytes,
            metal_allocated_bytes: inner.allocated_size() as u64,
            storage_mode: inner.storage_mode(),
            view_byte_offset: self.byte_offset,
            logical_bytes: self.data_byte_len,
            no_copy,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn descriptor_span_checks_actual_view_and_address_overflow() {
        assert_eq!(checked_span(100, 16, 8, 24).unwrap(), 116);
        assert!(checked_span(100, 17, 8, 24).is_err());
        assert!(checked_span(usize::MAX - 3, 0, 4, 4).is_err());
        assert!(checked_span(0, u64::MAX, 1, usize::MAX).is_err());
    }
    #[test]
    #[ignore = "fresh-process <=64KiB root backing metadata control; root supervised only"]
    fn actual_alias_subview_independent_owned_and_nocopy_descriptors() {
        use crate::{DType, MlxDevice, NativeRootAuthority, RootCreationLimits};
        use std::sync::Arc;
        let authority = NativeRootAuthority::install_early(RootCreationLimits {
            max_attempted_requested_bytes: 65536,
            max_root_attempts: 8,
            max_raw_constructor_attempts: 8,
            max_registered_devices: 1,
            max_rope_cache_entries: 1,
        })
        .unwrap();
        let device = MlxDevice::new().unwrap();
        authority.register_device(&device).unwrap();
        let a = device
            .upload_shared_bytes(&[9; 32], DType::U8, vec![32])
            .unwrap();
        let b = device
            .upload_shared_bytes(&[9; 32], DType::U8, vec![32])
            .unwrap();
        let view = a.data_view(8, 16, DType::U8, vec![16]).unwrap();
        let sa = a.backing_snapshot().unwrap();
        assert_eq!(sa.provenance, BufferProvenance::Owned);
        assert_eq!(sa, a.clone().backing_snapshot().unwrap());
        let sv = view.backing_snapshot().unwrap();
        assert_eq!(sv.origin, sa.origin);
        assert_eq!(sv.metal_root_bytes, sa.metal_root_bytes);
        assert_eq!((sv.view_byte_offset, sv.logical_bytes), (8, 16));
        assert_ne!(sa.origin, b.backing_snapshot().unwrap().origin);
        assert_eq!(a.as_slice::<u8>().unwrap(), b.as_slice::<u8>().unwrap());
        extern "C" {
            fn getpagesize() -> std::ffi::c_int;
        }
        let page = unsafe { getpagesize() };
        assert!(page > 0 && page <= 65536 - 64);
        let page = page as usize;
        let path =
            std::env::temp_dir().join(format!("mlx-backing-descriptor-{}", std::process::id()));
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        struct Cleanup(std::path::PathBuf);
        impl Drop for Cleanup {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(&self.0);
            }
        }
        let _cleanup = Cleanup(path);
        file.set_len(page as u64).unwrap();
        let mapping =
            Arc::new(unsafe { memmap2::MmapOptions::new().len(page).map(&file) }.unwrap());
        let n = device
            .map_file_buffer(mapping.clone(), 8, 16, DType::U8, vec![16])
            .unwrap();
        let sn = n.backing_snapshot().unwrap();
        assert_eq!(sn.provenance, BufferProvenance::NoCopy);
        let range = sn.no_copy.unwrap();
        assert_eq!(range.mapping_base_address, mapping.as_ptr() as usize);
        assert_eq!(range.mapping_bytes, page);
        assert_eq!(range.registered_base_address, mapping.as_ptr() as usize);
        assert_eq!(range.registered_bytes, page);
        assert_eq!(sn.view_byte_offset, 8);
        let nv = n.data_view(4, 8, DType::U8, vec![8]).unwrap();
        drop(n);
        assert_eq!(nv.backing_snapshot().unwrap().no_copy, Some(range));
        assert_eq!(nv.backing_snapshot().unwrap().view_byte_offset, 12);
        // An external wrapper remains unknown; observation cannot certify it.
        let imported = MlxBuffer::from_raw(b.storage.inner.clone(), DType::U8, vec![32]);
        assert_eq!(
            imported.backing_snapshot().unwrap().provenance,
            BufferProvenance::Unknown
        );
        println!("ROOT_BACKING_DESCRIPTOR_CONTROL alias=true subview=true independent=true nocopy=true import_unknown=true physical_credit_bytes=0 gpu_work=false");
    }
}
