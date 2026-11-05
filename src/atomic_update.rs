//! Conditional atomic read-modify-write without `fetch_update`.
//!
//! `fetch_update` is deprecated in favour of `try_update` on current stable
//! Rust, which the crate's MSRV (1.82) does not have. These helpers are the
//! same compare-exchange-weak loop with the same orderings and result:
//! `Ok(previous)` once `f` produced a value that was stored, `Err(previous)`
//! as soon as `f` returns `None`.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

macro_rules! define_try_update {
    ($name:ident, $atomic:ty, $value:ty) => {
        pub(crate) fn $name(
            atomic: &$atomic,
            set_order: Ordering,
            fetch_order: Ordering,
            mut f: impl FnMut($value) -> Option<$value>,
        ) -> Result<$value, $value> {
            let mut previous = atomic.load(fetch_order);
            while let Some(next) = f(previous) {
                match atomic.compare_exchange_weak(previous, next, set_order, fetch_order) {
                    Ok(stored) => return Ok(stored),
                    Err(current) => previous = current,
                }
            }
            Err(previous)
        }
    };
}

define_try_update!(try_update_u64, AtomicU64, u64);
define_try_update!(try_update_usize, AtomicUsize, usize);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stores_until_the_closure_refuses() {
        let counter = AtomicU64::new(1);
        let limit = |n: u64| n.checked_add(1).filter(|n| *n <= 2);
        assert_eq!(
            try_update_u64(&counter, Ordering::AcqRel, Ordering::Acquire, limit),
            Ok(1)
        );
        assert_eq!(
            try_update_u64(&counter, Ordering::AcqRel, Ordering::Acquire, limit),
            Err(2)
        );
        assert_eq!(counter.load(Ordering::Acquire), 2);

        let ticks = AtomicUsize::new(1);
        let down = |n: usize| n.checked_sub(1);
        assert_eq!(
            try_update_usize(&ticks, Ordering::AcqRel, Ordering::Acquire, down),
            Ok(1)
        );
        assert_eq!(
            try_update_usize(&ticks, Ordering::AcqRel, Ordering::Acquire, down),
            Err(0)
        );
        assert_eq!(ticks.load(Ordering::Acquire), 0);
    }
}
