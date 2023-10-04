//! Groups of attacks which are using artificially generated blocks
//! to investigate the effect on the replay stage performance.

use {
    crate::{accounts_file::AccountsFile, block_generator_config::BlockGeneratorConfig},
    solana_compute_budget::compute_budget_limits::MAX_COMPUTE_UNIT_LIMIT,
    std::sync::Arc,
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
    pub fn validate(&self, config: &Option<BlockGeneratorConfig>) -> Result<(), String> {
        const MAX_TRANSACTIONS_PER_BATCH: usize = 64;
        let Some(ref selected_attack) = self.selected_attack else {
            return Ok(());
        };
        let Some(config) = config else {
            return Err(
                "Cannot launch attack: accounts configuration file was not setup up".to_string(),
            );
        };
        let accounts = Arc::<AccountsFile>::from(config.clone().accounts);
        match selected_attack {
            Attack::CreateNonceAccounts => {
                if accounts.payers.len() < MAX_TRANSACTIONS_PER_BATCH {
                    return Err(format!(
                        "Not enough accounts for create nonce account generator: required at \
                         least {MAX_TRANSACTIONS_PER_BATCH}"
                    ));
                }
            }
            Attack::TransferRandom => {
                if accounts.payers.len() < 2 * MAX_TRANSACTIONS_PER_BATCH {
                    return Err(format!(
                        "Not enough accounts for random transfer generator: required at least {}",
                        2 * MAX_TRANSACTIONS_PER_BATCH
                    ));
                }
            }
            Attack::WriteProgram(attack) => {
                if accounts.owner_program_id.is_none() {
                    return Err(
                        "Accounts owner program is not specified. Cannot generate write program \
                         transactions."
                            .to_string(),
                    );
                }
                let accounts_batch_size =
                    attack.transaction_batch_size * attack.num_accounts_per_tx;
                if accounts.max_size.len() < accounts_batch_size {
                    return Err(format!(
                        "Accounts batch size {accounts_batch_size} is greater than the number of \
                         accounts provided"
                    ));
                }
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
            _ => (),
        }
        Ok(())
    }
}
