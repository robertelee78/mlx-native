//! Env-flag helpers for mlx-native feature gates.
//!
//! Mirrors `env_default_true` semantics from
//! `hf2q/src/debug/investigation_env.rs:1106-1121` so dispatcher gates in
//! `ops/*.rs` can flip from default-OFF/opt-in to default-ON/opt-out
//! without re-implementing the case-insensitive `0/false/off` parse at
//! every call site.
//!
//! ADR-028 §iter-326: parity-proven kernel ports default-flipped.

/// Default-ON boolean: returns `true` when the env var is unset OR set to
/// a truthy value (`"1"`, `"true"`, `"on"`, case-insensitive); returns
/// `false` only when explicitly set to a falsy value (`"0"`, `"false"`,
/// `"off"`, case-insensitive). Any other non-empty unrecognized value is
/// treated as `true` (permissive default-on: if someone sets a var they
/// probably want it on).
///
/// Used for feature flags that are default-ON and opt-out via
/// `=0` / `=false` / `=off`.
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
