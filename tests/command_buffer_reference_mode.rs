//! Host-only cold-process contract; never constructs a device or encoder.
use mlx_native::command_buffer_uses_unretained_references;
use std::{
    process::{Command, Stdio},
    time::{Duration, Instant},
};

const CHILD: &str = "MLX_NATIVE_REFERENCE_MODE_TEST_CHILD";

#[test]
#[ignore = "invoked only by cold_process_modes_are_stable"]
fn cold_process_child() {
    let expected = match std::env::var(CHILD).as_deref() {
        Ok("retained") => false,
        Ok("unretained") => true,
        _ => panic!("explicit private child mode required"),
    };
    assert_eq!(command_buffer_uses_unretained_references(), expected);
    // Mutation is confined to a fresh, single-test subprocess. The actual
    // cached native authority must not follow this later environment change.
    std::env::set_var("MLX_UNRETAINED_REFS", if expected { "0" } else { "1" });
    assert_eq!(command_buffer_uses_unretained_references(), expected);
    std::env::remove_var("MLX_UNRETAINED_REFS");
    assert_eq!(command_buffer_uses_unretained_references(), expected);
    let observed = std::thread::spawn(command_buffer_uses_unretained_references)
        .join()
        .expect("host accessor thread");
    assert_eq!(observed, expected);
}

#[test]
fn cold_process_modes_are_stable() {
    for (initial, expected) in [
        (None, false),
        (Some("0"), false),
        (Some("1"), true),
        (Some("true"), false),
        (Some("01"), false),
    ] {
        let mut command = Command::new(std::env::current_exe().expect("test executable"));
        command
            .args([
                "--exact",
                "cold_process_child",
                "--ignored",
                "--test-threads=1",
            ])
            .env(CHILD, if expected { "unretained" } else { "retained" })
            .env_remove("MLX_UNRETAINED_REFS")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit());
        if let Some(value) = initial {
            command.env("MLX_UNRETAINED_REFS", value);
        }
        let mut child = command.spawn().expect("spawn isolated host control");
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(status) = child.try_wait().expect("query host child") {
                assert!(status.success(), "initial mode {initial:?}: {status}");
                break;
            }
            if Instant::now() >= deadline {
                child.kill().expect("kill expired host child");
                child.wait().expect("reap expired host child");
                panic!("cold-process host control exceeded five seconds");
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}
