//! Observer-only ownership of actually submitted, not-yet-checked command buffers.
//! Never waits or infers status from another CB. Call only after an existing wait.
use super::*;
use std::sync::Mutex;
pub(super) const LIMIT: u64 = 1024;

pub(super) trait TerminalResource {
    fn status(&self) -> u64;
    fn gpu_timestamps(&self) -> crate::CompletedGpuTimestamps {
        crate::CompletedGpuTimestamps::unknown(crate::GpuTimestampUnknown::NotHealthyCompleted, None)
    }
}
struct Slot(Arc<AtomicU64>);
impl Drop for Slot {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}
struct Retired<R> {
    // These owners fully drop before the metadata/count charges below.
    resource: R,
    observation: CommandObservation,
    _metadata: Reservation,
    _slot: Slot,
}
pub(super) struct Registry<R> {
    items: Mutex<Vec<Retired<R>>>,
    outstanding: Arc<AtomicU64>,
    limit: u64,
}
impl<R: TerminalResource> Registry<R> {
    pub(super) fn new(limit: u64) -> Self {
        assert!(limit > 0 && limit <= LIMIT);
        Self {
            items: Mutex::new(Vec::new()),
            outstanding: Arc::new(AtomicU64::new(0)),
            limit,
        }
    }
    pub(super) fn outstanding(&self) -> u64 {
        self.outstanding.load(Ordering::Acquire)
    }
    pub(super) fn retain(&self, observation: CommandObservation, make: impl FnOnce() -> R) {
        let core = observation.core.clone();
        if std::thread::panicking() {
            core.fail();
            return;
        }
        if self
            .outstanding
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                n.checked_add(1).filter(|n| *n <= self.limit)
            })
            .is_err()
        {
            core.fail();
            // Outside any registry lock: unchecked observation Drop records Unknown.
            return;
        }
        let slot = Slot(self.outstanding.clone());
        let bytes = std::mem::size_of::<Retired<R>>() + std::mem::size_of::<usize>();
        let Ok(metadata) = reserve(&core, bytes) else {
            return;
        };
        // Resource retains happen only after both finite count and metadata charges.
        let mut entry = Some(Retired {
            resource: make(),
            observation,
            _metadata: metadata,
            _slot: slot,
        });
        let inserted = {
            match self.items.try_lock() {
                Ok(mut items) => {
                    if items.try_reserve_exact(1).is_err() {
                        false
                    } else {
                        items.push(entry.take().expect("owned pending entry"));
                        true
                    }
                }
                Err(_) => false,
            }
        };
        if !inserted {
            core.fail();
        }
        // No sink callback, ObjC release or observation Drop while holding items.
        drop(entry);
    }
    pub(super) fn inspect_after_wait(&self, core: &Arc<Core>, queue: u64) {
        // Concurrent submissions beyond this finite snapshot wait for a later drain.
        let watermark = core.next_id.load(Ordering::Acquire);
        for _ in 0..self.limit {
            let terminal = {
                let Ok(mut items) = self.items.try_lock() else {
                    core.fail();
                    return;
                };
                let found = items.iter().enumerate().find_map(|(i, entry)| {
                    if entry.observation.queue != queue || entry.observation.id >= watermark {
                        return None;
                    }
                    let begin = (core.clock)();
                    let status = entry.resource.status(); // Actual CB, not anchor status.
                    let end = (core.clock)();
                    matches!(status, 4 | 5).then_some((i, status, begin, end))
                });
                found.map(|(i, status, begin, end)| (items.swap_remove(i), status, begin, end))
            };
            let Some((entry, status, begin, end)) = terminal else {
                return;
            };
            // Preserve actual terminal Error evidence before sticky invalidation
            // suppresses future sink events. Error can never become healthy.
            entry.observation.drained(begin, end, status);
            entry.observation.completed_gpu_timing(|| entry.resource.gpu_timestamps());
            if status != 4 {
                core.fail();
            }
            // Fully release outside lock, then count/metadata refund. Nonterminal
            // resources stay retained; no sleep, poll loop, callback or extra wait.
            drop(entry);
        }
    }
}

pub(super) struct MetalCommand {
    command: Option<metal::CommandBuffer>,
    _queue: Option<metal::CommandQueue>,
}
impl TerminalResource for MetalCommand {
    fn gpu_timestamps(&self) -> crate::CompletedGpuTimestamps {
        crate::gpu_timestamps::read(self.command.as_ref().expect("retained actual CB"))
    }

    fn status(&self) -> u64 {
        self.command.as_ref().expect("retained actual CB").status() as u64
    }
}
impl Drop for MetalCommand {
    fn drop(&mut self) {
        // Labels/driver resources may need an autorelease pool on actual release.
        objc::rc::autoreleasepool(|| {
            drop(self.command.take());
            drop(self._queue.take());
        });
    }
}
pub(crate) fn retire(
    observation: CommandObservation,
    queue: &metal::CommandQueue,
    command: &metal::CommandBuffer,
) {
    if !observation.unchecked_work() {
        drop(observation);
        return;
    }
    let core = observation.core.clone();
    core.pending.retain(observation, || MetalCommand {
        command: Some(command.to_owned()),
        _queue: Some(queue.to_owned()),
    });
}
pub(crate) fn inspect_after_wait(queue: u64) {
    if let Some(core) = CORE.get() {
        core.pending.inspect_after_wait(core, queue);
    }
}

#[cfg(test)]
#[path = "pending_tests.rs"]
mod tests;
