use serde::{Deserialize, Serialize};

#[derive(Default, Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NonExistentAccountsAttackConfig {
    pub use_failed_transaction_hotpath: bool,
    pub use_invalid_fee_payer: bool,
}
