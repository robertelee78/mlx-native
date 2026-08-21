//! Persistent encoder worker thread (ADR-028 iter-380).
//!
//! Provides a long-lived worker thread for parallel command-buffer encoding,
//! mirroring the reference `n_cb=2` GCD `dispatch_apply` pattern.
//!
//! Per the existing `forward_decode` comment at line 4592-4595:
//! > Threaded wait DURING encode: -43 tok/s (thread spawn + Metal
//! > cross-thread synchronization overhead on command queue)
//!
//! That falsified attempt used per-token `std::thread::spawn`, paying the
//! ~50 µs spawn cost on every decode token.  This module amortizes that cost
//! by spawning the worker ONCE at process start, then submitting work via a
//! crossbeam-style mpsc channel.
//!
//! # Usage
//! ```ignore
//! use mlx_native::encoder_worker::EncoderWorker;
//!
//! // At process start (e.g., model load):
//! let worker = EncoderWorker::spawn();
//!
//! // Per-token (or per-encoding-task):
//! let (done_tx, done_rx) = std::sync::mpsc::channel();
//! worker.submit(move || {
//!     // ... encode work into a fresh CommandEncoder ...
//!     done_tx.send(()).ok();
//! });
//!
//! // Main thread can do its own work in parallel.
//!
//! // Eventually wait for worker to finish:
//! done_rx.recv().expect("worker died");
//! ```
//!
//! # Safety / lifetime
//!
//! - The worker thread is detached on `EncoderWorker::shutdown()` only.  The
//!   thread holds a `Receiver<Closure>`; when all `Sender` clones drop, the
//!   `iter()` loop exits naturally and the thread joins.
//! - Closures must be `'static` (they cross thread boundaries).  Use `Arc`
//!   for shared state.
//! - Closures must be `Send` (Rust's `mpsc::channel` enforces this).

use std::sync::mpsc;
use std::thread;

/// A submitted encoding task.  Boxed FnOnce because each task may capture
/// different types.  `Send + 'static` is required so the closure can be moved
/// to the worker thread.
type Task = Box<dyn FnOnce() + Send + 'static>;

/// A persistent worker thread that executes submitted closures sequentially
/// (in submission order).  Designed for command-buffer encoding workloads
/// where the cost of `std::thread::spawn` per task would dwarf the work.
///
/// The worker is single-threaded; submissions execute one-at-a-time.  For
/// parallelism with the main thread, the typical pattern is:
///
/// 1. Spawn one `EncoderWorker` at process start.
/// 2. Per token: submit half the encoding work to the worker, encode the
///    other half on the main thread, wait for both to complete.
///
/// `EncoderWorker` is NOT a thread pool — for that, spawn multiple workers.
pub struct EncoderWorker {
    tx: Option<mpsc::Sender<Task>>,
    handle: Option<thread::JoinHandle<()>>,
}

impl EncoderWorker {
    /// Spawn a new persistent worker thread.  The thread runs until either
    /// [`Self::shutdown`] is called or the `EncoderWorker` is dropped.
    ///
    /// The worker's run-loop blocks on the channel; CPU usage is zero when
    /// idle.
    pub fn spawn() -> Self {
        let (tx, rx) = mpsc::channel::<Task>();
        let handle = thread::Builder::new()
            .name("mlx-native-encoder-worker".into())
            .spawn(move || {
                // Run loop: pull tasks until the channel is closed.
                while let Ok(task) = rx.recv() {
                    task();
                }
            })
            .expect("failed to spawn encoder worker thread");
        Self { tx: Some(tx), handle: Some(handle) }
    }

    /// Submit a closure for execution on the worker thread.  Returns
    /// immediately; the closure runs asynchronously.
    ///
    /// To wait for the closure to complete, the caller must arrange its own
    /// signaling (e.g., a `(tx, rx)` channel pair captured by the closure).
    ///
    /// # Errors
    /// Returns `Err` if the worker thread has been shut down or has panicked.
    pub fn submit<F>(&self, f: F) -> Result<(), &'static str>
    where
        F: FnOnce() + Send + 'static,
    {
        match self.tx.as_ref() {
            Some(tx) => tx.send(Box::new(f)).map_err(|_| "worker thread is dead"),
            None => Err("worker has been shut down"),
        }
    }

    /// Cleanly shut down the worker.  Drops the sender (closing the channel),
    /// then joins the worker thread.  Returns once the worker has processed
    /// all in-flight tasks.
    pub fn shutdown(&mut self) {
        // Drop sender → channel closes → worker's recv() returns Err → loop exits.
        self.tx = None;
        if let Some(h) = self.handle.take() {
            // Ignore worker-panic errors during shutdown (already shutting down).
            let _ = h.join();
        }
    }
}

impl Drop for EncoderWorker {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[test]
    fn submit_runs_closure() {
        let worker = EncoderWorker::spawn();
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = Arc::clone(&counter);

        let (done_tx, done_rx) = std::sync::mpsc::channel();
        worker.submit(move || {
            counter_clone.fetch_add(1, Ordering::SeqCst);
            done_tx.send(()).ok();
        }).expect("submit");

        done_rx.recv().expect("worker did not signal completion");
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn submissions_run_in_order() {
        let worker = EncoderWorker::spawn();
        let order = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut signals = Vec::new();

        for i in 0..5 {
            let order_clone = Arc::clone(&order);
            let (tx, rx) = std::sync::mpsc::channel();
            signals.push(rx);
            worker.submit(move || {
                order_clone.lock().expect("lock").push(i);
                tx.send(()).ok();
            }).expect("submit");
        }

        for rx in signals {
            rx.recv().expect("worker panicked");
        }

        let final_order = order.lock().expect("lock").clone();
        assert_eq!(final_order, vec![0, 1, 2, 3, 4],
            "tasks ran out of order: {:?}", final_order);
    }

    #[test]
    fn shutdown_waits_for_in_flight_work() {
        let mut worker = EncoderWorker::spawn();
        let counter = Arc::new(AtomicU32::new(0));

        for _ in 0..3 {
            let counter_clone = Arc::clone(&counter);
            worker.submit(move || {
                std::thread::sleep(std::time::Duration::from_millis(10));
                counter_clone.fetch_add(1, Ordering::SeqCst);
            }).expect("submit");
        }

        worker.shutdown();
        // After shutdown, all 3 tasks should have completed.
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn submit_after_shutdown_errors() {
        let mut worker = EncoderWorker::spawn();
        worker.shutdown();
        assert!(worker.submit(|| {}).is_err());
    }

    // ---------------------------------------------------------------------
    // Metal-dispatch integration tests (ADR-028 iter-381)
    // ---------------------------------------------------------------------

    #[cfg(target_vendor = "apple")]
    #[test]
    fn worker_can_create_metal_encoder_and_commit() {
        // Validates: MlxDevice can be cloned + Arc'd + sent to worker thread,
        // CommandEncoder can be created from worker thread, commit_and_wait
        // works from worker thread.
        let device = crate::MlxDevice::new().expect("MlxDevice");
        let device_arc = Arc::new(device);
        let worker = EncoderWorker::spawn();

        let device_clone = Arc::clone(&device_arc);
        let (done_tx, done_rx) = std::sync::mpsc::channel::<Result<(), String>>();
        worker.submit(move || {
            let result = (|| -> Result<(), String> {
                let mut enc = device_clone.command_encoder()
                    .map_err(|e| format!("enc create: {e}"))?;
                enc.commit_and_wait()
                    .map_err(|e| format!("commit_and_wait: {e}"))?;
                Ok(())
            })();
            done_tx.send(result).ok();
        }).expect("submit");

        let result = done_rx.recv().expect("worker died");
        assert!(result.is_ok(), "worker Metal encoder failed: {:?}", result);
    }

    #[cfg(target_vendor = "apple")]
    #[test]
    fn worker_can_dispatch_real_kernel_zero_buffer() {
        // Validates: worker thread can register a kernel, allocate a buffer,
        // dispatch a real Metal compute kernel, commit + wait, and the host
        // sees the GPU-modified buffer contents after worker completion.
        use crate::DType;
        use crate::ops::moe_dispatch::moe_zero_buffer_encode;

        let device = crate::MlxDevice::new().expect("MlxDevice");
        let device_arc = Arc::new(device);
        let worker = EncoderWorker::spawn();

        // Allocate a buffer initialized to 1.0; worker should zero it via GPU.
        const N: usize = 1024;
        let mut buf = device_arc
            .alloc_buffer(N * 4, DType::F32, vec![N])
            .expect("alloc");
        for v in buf.as_mut_slice::<f32>().expect("init slice").iter_mut() {
            *v = 1.0;
        }

        // Wrap buffer in Arc<Mutex> so worker can mutate via &mut.
        let buf_arc = Arc::new(std::sync::Mutex::new(buf));
        let device_clone = Arc::clone(&device_arc);
        let buf_clone = Arc::clone(&buf_arc);

        let (done_tx, done_rx) = std::sync::mpsc::channel::<Result<(), String>>();
        worker.submit(move || {
            let result = (|| -> Result<(), String> {
                let mut registry = crate::KernelRegistry::new();
                let mut enc = device_clone.command_encoder()
                    .map_err(|e| format!("enc: {e}"))?;
                let buf_guard = buf_clone.lock().expect("lock");
                moe_zero_buffer_encode(
                    &mut enc, &mut registry, device_clone.metal_device(),
                    &buf_guard, N,
                ).map_err(|e| format!("zero_buffer: {e}"))?;
                drop(buf_guard); // release before commit so commit doesn't deadlock on a re-lock
                enc.commit_and_wait().map_err(|e| format!("commit: {e}"))?;
                Ok(())
            })();
            done_tx.send(result).ok();
        }).expect("submit");

        done_rx.recv().expect("worker died").expect("worker error");

        let buf_guard = buf_arc.lock().expect("lock");
        let slice = buf_guard.as_slice::<f32>().expect("read");
        for (i, &v) in slice.iter().enumerate() {
            assert_eq!(v, 0.0, "element {i} not zeroed: {v}");
        }
    }
}
