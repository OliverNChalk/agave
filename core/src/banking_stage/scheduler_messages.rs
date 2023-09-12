use {
    crate::banking_stage::consumer::RetryableIndex,
    solana_clock::{Epoch, Slot},
    std::fmt::Display,
};

/// A unique identifier for a transaction batch.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct TransactionBatchId(u64);

impl TransactionBatchId {
    pub fn new(index: u64) -> Self {
        Self(index)
    }
}

impl Display for TransactionBatchId {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

pub type TransactionId = usize;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MaxAge {
    pub sanitized_epoch: Epoch,
    pub alt_invalidation_slot: Slot,
}

impl MaxAge {
    pub const MAX: Self = Self {
        sanitized_epoch: Epoch::MAX,
        alt_invalidation_slot: Slot::MAX,
    };
}

/// Message: [Scheduler -> Worker]
/// Transactions to be consumed (i.e. executed, recorded, and committed)
pub struct ConsumeWork<Tx> {
    pub batch_id: TransactionBatchId,
    pub ids: Vec<TransactionId>,
    pub transactions: Vec<Tx>,
    pub max_ages: Vec<MaxAge>,

    /// This transaction is known to fail.  It should not be executed, instead it should be directly
    /// recorded as failed.  If the transaction does not fail, setting this flag would cause
    /// divergence.
    ///
    /// This functionality is used by the adversarial transaction generators, for transactions that
    /// are known to stress the execution system.  By skipping their execution, the adversarial
    /// leader can avoid running them for itself.
    ///
    /// TODO It would be even more flexible to have an ability to provide an execution result
    /// directly, rather than just saying that it must fail.
    pub use_failed_transaction_hotpath: bool,
}

/// Message: [Worker -> Scheduler]
/// Processed transactions.
pub struct FinishedConsumeWork<Tx> {
    pub work: ConsumeWork<Tx>,
    pub retryable_indexes: Vec<RetryableIndex>,
}
