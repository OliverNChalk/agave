use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_SEQ_ID: AtomicU64 = AtomicU64::new(0);

#[inline]
#[must_use]
#[cfg(target_arch = "x86_64")]
pub fn rdtsc() -> u64 {
    unsafe { std::arch::x86_64::_rdtsc() }
}

#[inline]
#[must_use]
#[cfg(not(target_arch = "x86_64"))]
pub fn rdtsc() -> u64 {
    0
}

pub fn seq_id() -> SeqId {
    SeqId(NEXT_SEQ_ID.fetch_add(1, Ordering::Relaxed))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct SeqId(pub u64);

pub struct TelemetryStamper {
    tx: Option<shaq::Producer<TelemetryStamp>>,
}

impl TelemetryStamper {
    #[cfg(unix)]
    pub fn open(name: &str) -> Self {
        Self {
            tx: crate::unix::try_open(name),
        }
    }

    #[cfg(not(unix))]
    pub fn open(_: &str) -> Self {
        Self(None)
    }

    pub fn stamp(&mut self, stamp: TelemetryStamp) {
        if let Some(tx) = self.tx.as_mut() {
            // PERF: Need a diff queue.
            tx.sync();
            let _ = tx.try_write(stamp);
            tx.commit();
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct TelemetryStamp {
    /// Unique sequence number for this transaction.
    pub seq_id: SeqId,
    /// Raw CPU ticks this stamp was taken at.
    pub rdtsc: u64,
    /// Standardize action taken against `seq_id`.
    pub action: TelemetryAction,
    /// Free form code field.
    pub code: u32,
}

#[derive(Debug, Clone, Copy)]
#[repr(u32)]
pub enum TelemetryAction {
    Recv = 1,
    Send = 2,
    Drop = 3,
}
