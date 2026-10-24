use super::{BoundedPoolError as E, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Status {
    Issued(u64),
    Free,
    Retired,
}

/// Single membership record: release/reset cannot insert duplicate free roots.
pub(super) struct RootState {
    pub status: Status,
    epoch: u64,
}
impl RootState {
    pub fn fresh() -> Self {
        Self {
            status: Status::Issued(1),
            epoch: 1,
        }
    }
    pub fn issue(&mut self) -> Result<u64> {
        if self.status != Status::Free {
            return Err(E::InvalidLease);
        }
        let next = self.epoch.checked_add(1).ok_or(E::Overflow)?;
        self.epoch = next;
        self.status = Status::Issued(next);
        Ok(next)
    }
    pub fn release(&mut self, epoch: u64) -> Result<()> {
        if self.status != Status::Issued(epoch) {
            return Err(E::InvalidLease);
        }
        self.status = Status::Free;
        Ok(())
    }
    pub fn reset(&mut self) {
        if matches!(self.status, Status::Issued(_)) {
            self.status = Status::Free;
        }
    }
}

pub(super) fn bucket(bytes: usize) -> Result<usize> {
    if bytes == 0 {
        return Err(E::InvalidRequest);
    }
    bytes.checked_next_power_of_two().ok_or(E::Overflow)
}
pub(super) fn reset_allowed(live: usize) -> Result<()> {
    if live != 0 {
        Err(E::LiveLeases)
    } else {
        Ok(())
    }
}

#[cfg(test)]
#[test]
fn epoch_overflow_preserves_free_state() {
    let mut root = RootState {
        status: Status::Free,
        epoch: u64::MAX,
    };
    assert_eq!(root.issue(), Err(E::Overflow));
    assert_eq!(root.status, Status::Free);
    assert_eq!(root.epoch, u64::MAX);
}
