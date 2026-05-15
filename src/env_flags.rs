//! Env-flag helpers for mlx-native feature gates.
//!
//! Mirrors `env_default_true` semantics from
//! `hf2q/src/debug/investigation_env.rs:1106-1121` so dispatcher gates in
//! `ops/*.rs` can flip from default-OFF/opt-in to default-ON/opt-out
//! without re-implementing the case-insensitive `0/false/off` parse at
//! every call site.
//!
//! ADR-028 §iter-326: parity-proven kernel ports default-flipped.
//!
//! ADR-029 iter-175 Step 1an (2026-05-15): hot-path env reads now use
//! cached `q6k_id_mv_nr2_enabled` / `q6k_mv_nr2_enabled` / etc. helpers
//! that store the parsed value in an `AtomicI8` after first call.  H-N
//! microbench (mlx-native 2061fd8) found uncached `env_default_true` costs
//! ~69 ns/call; per-token total of ~1000 such calls = ~1% wall.  Cached
//! version costs ~2-3 ns/call.

use std::sync::atomic::{AtomicI8, Ordering};

/// Default-ON boolean: returns `true` when the env var is unset OR set to
/// a truthy value (`"1"`, `"true"`, `"on"`, case-insensitive); returns
/// `false` only when explicitly set to a falsy value (`"0"`, `"false"`,
/// `"off"`, case-insensitive). Any other non-empty unrecognized value is
/// treated as `true` (permissive default-on: if someone sets a var they
/// probably want it on).
///
/// Used for feature flags that are default-ON and opt-out via
/// `=0` / `=false` / `=off`.
///
/// **Performance**: this function calls `std::env::var(name)` each time,
/// which on Unix holds a mutex on the env array and does a linear scan.
/// Cost is ~70 ns/call. For hot-path dispatch gates that don't change
/// after process start, prefer one of the cached helpers below
/// (`q6k_id_mv_nr2_enabled`, `q6k_mv_nr2_enabled`, etc.) which amortize
/// the read into a single AtomicI8 load (~2 ns/call after first call).
#[inline]
pub(crate) fn env_default_true(name: &str) -> bool {
    match std::env::var(name).ok().as_deref() {
        // Env unset → default ON.
        None => true,
        // Falsy: "0", "false", "off" (case-insensitive) → OFF.
        Some(v)
            if v.eq_ignore_ascii_case("0")
                || v.eq_ignore_ascii_case("false")
                || v.eq_ignore_ascii_case("off") =>
        {
            false
        }
        // Truthy or unrecognized → ON (permissive).
        Some(_) => true,
    }
}

/// Cached env-flag read pattern: declare `static CACHED: AtomicI8 = AtomicI8::new(-1);`
/// then call `cached_env_default_true(&CACHED, "VAR_NAME")`.  The first call
/// reads + parses the env var (~70 ns); subsequent calls do a single atomic
/// load (~2 ns).  Use this pattern for flags that don't change after
/// process startup.
///
/// Returns the same value as `env_default_true(name)`.
#[inline]
pub(crate) fn cached_env_default_true(cache: &AtomicI8, name: &str) -> bool {
    let v = cache.load(Ordering::Relaxed);
    if v >= 0 {
        return v == 1;
    }
    let on = env_default_true(name);
    // Race here is benign: two threads may both write the same value.
    cache.store(if on { 1 } else { 0 }, Ordering::Relaxed);
    on
}

/// Cached version of `env::var(name).as_deref() == Ok("1")` — for opt-in
/// flags (default-OFF) that are turned on with `=1` exactly.
#[inline]
pub(crate) fn cached_env_eq_one(cache: &AtomicI8, name: &str) -> bool {
    let v = cache.load(Ordering::Relaxed);
    if v >= 0 {
        return v == 1;
    }
    let on = std::env::var(name).ok().as_deref() == Some("1");
    cache.store(if on { 1 } else { 0 }, Ordering::Relaxed);
    on
}

#[cfg(test)]
mod tests {
    use super::env_default_true;
    use std::sync::Mutex;

    // Serialize env access — Rust tests run threaded by default and
    // `set_var` is process-global.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn with_env<F: FnOnce()>(name: &str, value: Option<&str>, body: F) {
        let _g = ENV_LOCK.lock().expect("env lock poisoned");
        let prev = std::env::var(name).ok();
        match value {
            Some(v) => std::env::set_var(name, v),
            None => std::env::remove_var(name),
        }
        body();
        match prev {
            Some(v) => std::env::set_var(name, v),
            None => std::env::remove_var(name),
        }
    }

    #[test]
    fn unset_returns_true() {
        with_env("MLX_NATIVE_TEST_DEFAULT_TRUE_A", None, || {
            assert!(env_default_true("MLX_NATIVE_TEST_DEFAULT_TRUE_A"));
        });
    }

    #[test]
    fn explicit_one_returns_true() {
        with_env("MLX_NATIVE_TEST_DEFAULT_TRUE_B", Some("1"), || {
            assert!(env_default_true("MLX_NATIVE_TEST_DEFAULT_TRUE_B"));
        });
    }

    #[test]
    fn explicit_zero_returns_false() {
        with_env("MLX_NATIVE_TEST_DEFAULT_TRUE_C", Some("0"), || {
            assert!(!env_default_true("MLX_NATIVE_TEST_DEFAULT_TRUE_C"));
        });
    }

    #[test]
    fn explicit_false_returns_false() {
        with_env("MLX_NATIVE_TEST_DEFAULT_TRUE_D", Some("false"), || {
            assert!(!env_default_true("MLX_NATIVE_TEST_DEFAULT_TRUE_D"));
        });
        with_env("MLX_NATIVE_TEST_DEFAULT_TRUE_D", Some("FALSE"), || {
            assert!(!env_default_true("MLX_NATIVE_TEST_DEFAULT_TRUE_D"));
        });
    }

    #[test]
    fn explicit_off_returns_false() {
        with_env("MLX_NATIVE_TEST_DEFAULT_TRUE_E", Some("off"), || {
            assert!(!env_default_true("MLX_NATIVE_TEST_DEFAULT_TRUE_E"));
        });
        with_env("MLX_NATIVE_TEST_DEFAULT_TRUE_E", Some("OFF"), || {
            assert!(!env_default_true("MLX_NATIVE_TEST_DEFAULT_TRUE_E"));
        });
    }

    #[test]
    fn unrecognized_returns_true_permissive() {
        with_env("MLX_NATIVE_TEST_DEFAULT_TRUE_F", Some("yes"), || {
            assert!(env_default_true("MLX_NATIVE_TEST_DEFAULT_TRUE_F"));
        });
        with_env("MLX_NATIVE_TEST_DEFAULT_TRUE_F", Some("enable"), || {
            assert!(env_default_true("MLX_NATIVE_TEST_DEFAULT_TRUE_F"));
        });
    }
}
