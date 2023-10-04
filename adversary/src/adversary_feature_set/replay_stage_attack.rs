//! Groups of attacks which are using artificially generated blocks
//! to investigate the effect on the replay stage performance.

use {
    solana_compute_budget::compute_budget_limits::MAX_COMPUTE_UNIT_LIMIT,
    strum::VariantNames,
    strum_macros::{Display, EnumString, EnumVariantNames},
};

pub const ID: &str = "replay_stage_attack";
adversarial_feature_impl!(ReplayStageAttack);

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WriteProgramConfig {
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
impl Default for WriteProgramConfig {
    fn default() -> Self {
        Self {
            transaction_batch_size: 1,
            num_accounts_per_tx: 1,
            transaction_cu_budget: 1_000,
            use_failed_transaction_hotpath: false,
        }
    }
}

#[derive(
    Clone,
    Debug,
    Display,
    Eq,
    EnumString,
    EnumVariantNames,
    PartialEq,
    serde::Serialize,
    serde::Deserialize,
)]
#[serde(rename_all = "camelCase")]
#[strum(serialize_all = "camelCase")]
pub enum Attack {
    TransferRandom,
    CreateNonceAccounts,
    AllocateRandomLarge,
    AllocateRandomSmall,
    ChainTransactions,
    WriteProgram(WriteProgramConfig),
}

impl Attack {
    pub const fn cli_names() -> &'static [&'static str] {
        Self::VARIANTS
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AdversarialConfig {
    pub selected_attack: Option<Attack>,
}

impl AdversarialConfig {
    pub fn new(selected_attack: Option<Attack>) -> Result<Self, String> {
        let config = AdversarialConfig { selected_attack };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), String> {
        if let Some(Attack::WriteProgram(attack)) = &self.selected_attack {
            if attack.transaction_batch_size == 0 || attack.transaction_batch_size > 64 {
                return Err(format!(
                    "transaction_batch_size ({}) must be in range [1, 64]",
                    attack.transaction_batch_size
                ));
            }
            if attack.num_accounts_per_tx == 0 || attack.num_accounts_per_tx > 48 {
                return Err(format!(
                    "number of accounts per transactions ({}) must be in range [1, 48]",
                    attack.num_accounts_per_tx
                ));
            }
            if attack.transaction_cu_budget > MAX_COMPUTE_UNIT_LIMIT {
                return Err(format!(
                    "transaction_cu_budget ({}) is greater than max value ({})",
                    attack.transaction_cu_budget, MAX_COMPUTE_UNIT_LIMIT
                ));
            }
        }
        Ok(())
    }
}
