//! Submission-counter contract.
//!
//! `cmd_buf_count` records allocation while `commit_count` records the actual
//! Metal submission boundary.  Keeping those facts separate prevents
//! consumers from reporting allocated-but-never-submitted buffers as GPU work.

use mlx_native::{cmd_buf_count, commit_count, reset_counters, sync_count, MlxDevice};

#[test]
fn creation_submission_and_wait_are_distinct() {
    let device = MlxDevice::new().expect("create Metal device");
    reset_counters();

    let mut async_encoder = device.command_encoder().expect("create async encoder");
    assert_eq!(cmd_buf_count(), 1, "creating an encoder allocates one CB");
    assert_eq!(commit_count(), 0, "allocation is not a submission");
    assert_eq!(sync_count(), 0, "allocation is not a CPU wait");

    async_encoder.commit();
    assert_eq!(commit_count(), 1, "async commit submits exactly once");
    assert_eq!(sync_count(), 0, "async commit does not wait");

    let mut sync_encoder = device.command_encoder().expect("create sync encoder");
    assert_eq!(cmd_buf_count(), 2);
    sync_encoder
        .commit_and_wait()
        .expect("submit and drain both same-queue command buffers");

    assert_eq!(commit_count(), 2, "both command buffers were submitted");
    assert_eq!(sync_count(), 1, "only the synchronous commit waited");
}
