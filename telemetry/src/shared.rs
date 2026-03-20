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
    stage: TelemetryStage,
    tx: Option<shaq::Producer<TelemetryStamp>>,
}

impl TelemetryStamper {
    #[cfg(unix)]
    pub fn open(stage: TelemetryStage) -> Self {
        Self {
            stage,
            tx: crate::unix::try_open(stage.into()),
        }
    }

    #[cfg(not(unix))]
    pub fn open(_: &str) -> Self {
        Self(None)
    }

    pub fn ingest(&mut self) -> SeqId {
        let seq_id = crate::seq_id();
        self.stamp(seq_id, TelemetryAction::Recv);

        seq_id
    }

    pub fn stamp(&mut self, seq_id: SeqId, action: TelemetryAction) {
        if let Some(tx) = self.tx.as_mut() {
            // PERF: Need a diff queue.
            tx.sync();
            let _ = tx.try_write(TelemetryStamp {
                seq_id,
                rdtsc: crate::rdtsc(),
                stage: self.stage,
                action,
            });
            tx.commit();
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct TelemetryStamp {
    pub seq_id: SeqId,
    pub rdtsc: u64,
    pub stage: TelemetryStage,
    pub action: TelemetryAction,
}

#[derive(Debug, Clone, Copy, strum::IntoStaticStr)]
#[strum(serialize_all = "kebab-case")]
#[repr(u32)]
pub enum TelemetryStage {
    TpuToPack = 1,
}

#[derive(Debug, Clone, Copy)]
#[repr(u32)]
pub enum TelemetryAction {
    Recv = 1,
    Send = 2,
    Drop = 3,
}
