use serde::{Deserialize, Serialize};

/// Configuration used by attacks that invoke the "attack program".
///
/// Specifically these would be `WriteProgram`, `ReadProgram`, `RecursiveProgram`, and
/// `ColdProgramCache` attacks.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AttackProgramConfig {
    /// Max value is 64. In some parts of the code it is called "entry size".
    pub transaction_batch_size: usize,
    pub num_accounts_per_tx: usize,
    pub transaction_cu_budget: u32,
    /// Attacks involving expensive computations might be configured with
    /// option to bypass execution. For that, they must be configured to fail.
    /// This might be achieved by requesting less `transaction_cu_budget` than required
    pub use_failed_transaction_hotpath: bool,
}

// Default values are such that generated block can be replayed in ~400ms.
// Generating heavier blocks is possible but requires skipping loading accounts and execution
// transactions in the block.
impl Default for AttackProgramConfig {
    fn default() -> Self {
        Self {
            transaction_batch_size: 1,
            num_accounts_per_tx: 1,
            // high enough value so that transaction is valid
            transaction_cu_budget: 10_000,
            use_failed_transaction_hotpath: false,
        }
    }
}
