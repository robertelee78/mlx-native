use std::sync::atomic::{AtomicUsize, Ordering};

/// The payload must finish destruction before its metadata slot is reusable.
/// Taking it first prevents automatic field Drop from occurring after release.
/// If destruction panics, the decrement is never reached: fail closed.
pub(super) fn drop_payload_then_release<T>(payload: Option<T>, live: &AtomicUsize) {
    drop(payload.expect("live lease payload"));
    live.fetch_sub(1, Ordering::AcqRel);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{mpsc, Arc};
    use std::time::Duration;

    struct PausedPayload {
        entered: mpsc::Sender<()>,
        release: mpsc::Receiver<()>,
    }
    impl Drop for PausedPayload {
        fn drop(&mut self) {
            self.entered.send(()).unwrap();
            self.release.recv().unwrap();
        }
    }

    #[test]
    fn paused_payload_keeps_charge_until_destruction_returns() {
        let live = Arc::new(AtomicUsize::new(1));
        let worker_live = live.clone();
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            drop_payload_then_release(
                Some(PausedPayload {
                    entered: entered_tx,
                    release: release_rx,
                }),
                &worker_live,
            );
        });
        let entered = entered_rx.recv_timeout(Duration::from_secs(5));
        let during_destruction = live.load(Ordering::Acquire);
        let reset_during_destruction =
            crate::bounded_pool::state::reset_allowed(during_destruction);
        // Always release/join before assertions, including a failed entry signal.
        let release = release_tx.send(());
        let joined = worker.join();
        let after_destruction = live.load(Ordering::Acquire);
        assert!(entered.is_ok(), "destructor did not enter its pause");
        assert!(release.is_ok(), "destructor release channel failed");
        assert!(joined.is_ok(), "worker destructor failed");
        assert_eq!(during_destruction, 1);
        assert_eq!(
            reset_during_destruction,
            Err(crate::bounded_pool::BoundedPoolError::LiveLeases)
        );
        assert_eq!(after_destruction, 0);
        assert_eq!(
            crate::bounded_pool::state::reset_allowed(after_destruction),
            Ok(())
        );
    }

    #[test]
    fn panicking_payload_keeps_charge_fail_closed() {
        struct PanickingPayload;
        impl Drop for PanickingPayload {
            fn drop(&mut self) {
                panic!("controlled payload destructor panic");
            }
        }
        let live = AtomicUsize::new(1);
        let result = std::panic::catch_unwind(|| {
            drop_payload_then_release(Some(PanickingPayload), &live);
        });
        assert!(result.is_err());
        assert_eq!(live.load(Ordering::Acquire), 1);
    }
}
