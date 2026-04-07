//! Data-type enumeration for tensor elements.

use std::fmt;

/// Element data type carried by an [`MlxBuffer`](crate::MlxBuffer).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// 32-bit IEEE 754 float.
    F32,
    /// 16-bit IEEE 754 half-precision float.
    F16,
    /// 16-bit brain floating point.
    BF16,
    /// Unsigned 8-bit integer.
    U8,
    /// Unsigned 16-bit integer.
    U16,
    /// Unsigned 32-bit integer.
    U32,
    /// Signed 32-bit integer.
    I32,
}

impl DType {
    /// Size of a single element in bytes.
    #[inline]
    pub fn size_of(self) -> usize {
        match self {
            DType::F32 | DType::U32 | DType::I32 => 4,
            DType::F16 | DType::BF16 | DType::U16 => 2,
            DType::U8 => 1,
        }
    }

    /// Short lowercase name, e.g. `"f32"`, `"bf16"`.
    #[inline]
    pub fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::U8 => "u8",
            DType::U16 => "u16",
            DType::U32 => "u32",
            DType::I32 => "i32",
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_size_of() {
        assert_eq!(DType::F32.size_of(), 4);
        assert_eq!(DType::F16.size_of(), 2);
        assert_eq!(DType::BF16.size_of(), 2);
        assert_eq!(DType::U8.size_of(), 1);
        assert_eq!(DType::U16.size_of(), 2);
        assert_eq!(DType::U32.size_of(), 4);
        assert_eq!(DType::I32.size_of(), 4);
    }

    #[test]
    fn test_dtype_display() {
        assert_eq!(format!("{}", DType::F32), "f32");
        assert_eq!(format!("{}", DType::BF16), "bf16");
        assert_eq!(format!("{}", DType::U8), "u8");
    }
}
