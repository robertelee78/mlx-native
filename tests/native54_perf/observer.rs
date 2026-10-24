//! Installed SDK: NSInteger thermalState; platform objc::runtime::BOOL getters.
use objc::runtime::{Class, Object, BOOL, NO};
use objc::{msg_send, sel, sel_impl};
use std::time::Instant;
#[link(name = "Foundation", kind = "framework")]
extern "C" {}
#[derive(Clone, Copy)]
pub struct Observation {
    pub elapsed: f64,
    pub thermal: isize,
    pub low_power: i8,
    pub available: bool,
}
impl Observation {
    pub const EMPTY: Self = Self {
        elapsed: 0.0,
        thermal: -1,
        low_power: -1,
        available: false,
    };
    pub fn qualified(self) -> bool {
        self.available && self.thermal == 0 && self.low_power == 0
    }
}
pub fn sample(epoch: Instant) -> Observation {
    let mut result = Observation {
        elapsed: epoch.elapsed().as_secs_f64(),
        ..Observation::EMPTY
    };
    let Some(class) = Class::get("NSProcessInfo") else {
        return result;
    };
    // Availability checked before each message. SDK NSProcessInfoThermalState
    // is NS_ENUM(NSInteger), isize on macOS; BOOL uses objc's target-specific ABI
    // (bool on aarch64, signed char on x86_64), not an assumed Rust integer.
    unsafe {
        let responds: BOOL = msg_send![class, respondsToSelector:sel!(processInfo)];
        if responds == NO {
            return result;
        }
        let info: *mut Object = msg_send![class, processInfo];
        if info.is_null() {
            return result;
        }
        let thermal: BOOL = msg_send![info,respondsToSelector:sel!(thermalState)];
        let power: BOOL = msg_send![info,respondsToSelector:sel!(isLowPowerModeEnabled)];
        if thermal == NO || power == NO {
            return result;
        }
        result.thermal = msg_send![info, thermalState];
        let low: BOOL = msg_send![info, isLowPowerModeEnabled];
        result.low_power = if low == NO { 0 } else { 1 };
        result.available = (0..=3).contains(&result.thermal);
    }
    result
}
