//! Generators for testing banking stage

use {
    solana_adversary::{
        accounts_file::AccountsFile, adversary_feature_set::replay_stage_attack::Attack,
        block_generator_config::BlockGeneratorConfig,
    },
    solana_runtime::bank::Bank,
    solana_transaction::sanitized::SanitizedTransaction,
    std::sync::Arc,
};

pub(super) mod allocate_random_large;
pub(super) mod allocate_random_small;
pub(super) mod chain_transactions;
pub(super) mod create_nonce_accounts;
pub(super) mod read_max_accounts;
pub(super) mod transfer_random;
pub(super) mod write_max_accounts;
pub(super) mod write_program;

/// Attacks are essentially sequences of transactions.  Each will have a function that is
/// responsible for generating these transactions, and those functions will adhere to this type.
pub type TransactionGenerator = Box<dyn Send + FnMut(&Bank) -> (Vec<SanitizedTransaction>, usize)>;

/// Encapsulate logic for managing selected generator
pub struct ActiveGenerator {
    num_workers: usize,
    accounts: Arc<AccountsFile>,
    current: Option<(TransactionGenerator, /* execution tx batches */ usize)>,
    // used to simplify check if the selected_generator should be changed
    current_attack: Option<Attack>,
}

impl ActiveGenerator {
    pub fn new(block_generator_config: BlockGeneratorConfig, num_workers: usize) -> Self {
        let accounts = Arc::<AccountsFile>::from(block_generator_config.accounts);
        Self {
            num_workers,
            accounts,
            current: None,
            current_attack: None,
        }
    }

    /// Run the generator this many times at once, when producing transactions.
    /// Generators that do little work per transaction should use a larger value.
    pub fn get_num_generator_exec_batch_size(&self) -> usize {
        self.current
            .as_ref()
            .map(|generator| generator.1)
            .unwrap_or(0)
    }

    /// Generate transactions using current generator
    pub fn generate_transactions(&mut self, bank: &Bank) -> (Vec<SanitizedTransaction>, usize) {
        self.current
            .as_mut()
            .map(|generator| (generator.0)(bank))
            .unwrap_or((vec![], 0))
    }

    fn create_generator(&self) -> Option<(TransactionGenerator, usize)> {
        let Self {
            accounts,
            num_workers,
            current_attack,
            ..
        } = self;
        let accounts = accounts.clone();

        use Attack::*;
        current_attack.as_ref().map(|attack| match attack {
            TransferRandom => (transfer_random::generator(accounts, *num_workers), 100),
            CreateNonceAccounts => (create_nonce_accounts::generator(accounts, *num_workers), 10),
            AllocateRandomLarge => (allocate_random_large::generator(accounts, *num_workers), 1),
            AllocateRandomSmall => (allocate_random_small::generator(accounts, *num_workers), 10),
            ChainTransactions => (chain_transactions::generator(accounts, *num_workers), 10),
            WriteProgram(write_program_config) => (
                write_program::generator(accounts, *num_workers, write_program_config.clone()),
                1,
            ),
            ReadMaxAccounts => (read_max_accounts::generator(accounts, *num_workers), 1),
            WriteMaxAccounts => (write_max_accounts::generator(accounts, *num_workers), 1),
        })
    }

    /// Create a new generator using existing accounts and reset the state of the structure
    pub fn ensure_active(&mut self, attack: Option<Attack>) {
        if self.current_attack == attack {
            return;
        }
        info!("Reset selected generator to: {attack:?}");
        self.current_attack = attack.clone();
        self.current = self.create_generator();
    }

    /// Check if any generator has been selected
    pub fn is_active(&self) -> bool {
        self.current.is_some()
    }

    /// Attacks involving expensive computations might be configured with
    /// option to bypass execution. For that, they must be configured to fail
    pub fn use_failed_transaction_hotpath(&self) -> bool {
        if let Some(Attack::WriteProgram(write_attack_config)) = &self.current_attack {
            write_attack_config.use_failed_transaction_hotpath
        } else {
            false
        }
    }
}
